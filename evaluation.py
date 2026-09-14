from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from graph import compiled_graph
from retriever import QueryEmbedder, dedupe_by_madde
from llm_utils import GroqDailyQuotaExceededError

warnings.filterwarnings("ignore")

REFUSAL_MARKERS = ("hukuk uzmanına danışın", "bulamadım", "yeterli bilgi yok")


@dataclass
class EvalCase:
    soru: str
    beklenen_kanun: Optional[str]
    beklenen_madde: List[str]
    kapsam_disi: bool


@dataclass
class EvalResult:
    soru: str
    cevap: str
    retrieval_hit: Optional[bool]
    dogru_reddetti: Optional[bool]
    is_grounded: bool
    retry_count: int
    sure_sn: float
    kaynaklar: List[str]
    hata: Optional[str] = None


def load_testset(path: str) -> List[EvalCase]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [EvalCase(**item) for item in raw]


def run_case(
    case: EvalCase, embedder: QueryEmbedder, top_k: int = 5, max_retries: int = 2
) -> EvalResult:
    t0 = time.perf_counter()
    initial_state = {
        "query": case.soru,
        "kanun_adi": None,
        "top_k": top_k,
        "embedder": embedder,
        "results": [],
        "answer": "",
        "is_grounded": False,
        "grading_reason": "",
        "retry_count": 0,
        "max_retries": max_retries,
    }
    final_state = compiled_graph.invoke(initial_state)
    sure = time.perf_counter() - t0

    retrieved_madde = {(r.kanun_adi, r.madde_no) for r in final_state["results"]}

    retrieval_hit = None
    if not case.kapsam_disi and case.beklenen_madde:
        retrieval_hit = any(
            (case.beklenen_kanun, m) in retrieved_madde for m in case.beklenen_madde
        )

    dogru_reddetti = None
    if case.kapsam_disi:
        answer_low = final_state["answer"].lower()
        dogru_reddetti = any(marker in answer_low for marker in REFUSAL_MARKERS)

    return EvalResult(
        soru=case.soru,
        cevap=final_state["answer"],
        retrieval_hit=retrieval_hit,
        dogru_reddetti=dogru_reddetti,
        is_grounded=final_state["is_grounded"],
        retry_count=final_state["retry_count"],
        sure_sn=round(sure, 2),
        kaynaklar=[r.context_path for r in dedupe_by_madde(final_state["results"])],
    )


def _failed_result(case: EvalCase, error: Exception, sure_sn: float) -> EvalResult:
    """run_case tamamen patladığında (ör. Groq 429->RuntimeError) koşuyu
    durdurmak yerine bu case'i 'hata' olarak işaretleyip devam etmemizi sağlar."""
    return EvalResult(
        soru=case.soru,
        cevap="",
        retrieval_hit=None,
        dogru_reddetti=None,
        is_grounded=False,
        retry_count=0,
        sure_sn=round(sure_sn, 2),
        kaynaklar=[],
        hata=str(error),
    )


def summarize(results: List[EvalResult]) -> dict:
    basarili = [r for r in results if r.hata is None]
    in_scope = [r for r in basarili if r.retrieval_hit is not None]
    out_scope = [r for r in basarili if r.dogru_reddetti is not None]
    hatali = [r for r in results if r.hata is not None]

    retrieval_accuracy = (
        sum(r.retrieval_hit for r in in_scope) / len(in_scope) if in_scope else None
    )
    refusal_accuracy = (
        sum(r.dogru_reddetti for r in out_scope) / len(out_scope)
        if out_scope
        else None
    )
    avg_time = sum(r.sure_sn for r in basarili) / len(basarili) if basarili else 0
    avg_retry = sum(r.retry_count for r in basarili) / len(basarili) if basarili else 0

    return {
        "toplam_soru": len(results),
        "basarili_soru": len(basarili),
        "hatali_soru": len(hatali),
        "kapsam_ici_soru": len(in_scope),
        "kapsam_disi_soru": len(out_scope),
        "retrieval_isabet_orani": retrieval_accuracy,
        "dogru_red_orani": refusal_accuracy,
        "ortalama_sure_sn": round(avg_time, 2),
        "ortalama_retry": round(avg_retry, 2),
    }


def _save(path: str, results: List[EvalResult]) -> None:
    """Her case sonrası çağrılır: bir çökme/kesinti olsa bile o ana kadarki
    tüm sonuçlar diskte kalır."""
    summary = summarize(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"ozet": summary, "detaylar": [asdict(r) for r in results]},
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", default="testset.json")
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Grounding başarısız olduğunda generate node'unun tekrar deneme sayısı.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=4.0,
        help="Groq rate limit'ini aşmamak için sorular arası bekleme (sn).",
    )
    args = parser.parse_args()

    cases = load_testset(args.testset)
    print(f"{len(cases)} test sorusu yüklendi, çalıştırılıyor...\n")

    embedder = QueryEmbedder()

    results: List[EvalResult] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.soru}")
        t0 = time.perf_counter()
        try:
            result = run_case(
                case, embedder, top_k=args.top_k, max_retries=args.max_retries
            )
        except GroqDailyQuotaExceededError as e:
            sure = time.perf_counter() - t0
            result = _failed_result(case, e, sure)
            results.append(result)
            _save(args.output, results)
            print(f"    GÜNLÜK KOTA DOLDU, çalıştırma durduruluyor: {e}")
            print(
                f"    {i}/{len(cases)} soru denendi, kalanlar atlandı. "
                f"Mevcut sonuçlar {args.output} dosyasında."
            )
            break
        except Exception as e:
            sure = time.perf_counter() - t0
            print(f"    HATA: {e}")
            result = _failed_result(case, e, sure)
        else:
            status_bits = []
            if result.retrieval_hit is not None:
                status_bits.append(
                    "retrieval:" + ("✅" if result.retrieval_hit else "❌")
                )
            if result.dogru_reddetti is not None:
                status_bits.append(
                    "red:" + ("✅" if result.dogru_reddetti else "❌")
                )
            print(
                f"    {' '.join(status_bits)} | grounded={result.is_grounded} "
                f"| {result.sure_sn}sn"
            )

        results.append(result)
        _save(args.output, results)

        time.sleep(args.sleep)

    summary = summarize(results)

    print("\n=== ÖZET ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nDetaylı sonuçlar {args.output} dosyasına yazıldı.")


if __name__ == "__main__":
    main()
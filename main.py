from __future__ import annotations
import sys

from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from retriever import QueryEmbedder, dedupe_by_madde
from graph import compiled_graph


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Kullanım: python main.py "soru metni" [kanun_adi]')

    query = sys.argv[1]
    kanun_adi = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Soru: {query}\n")

    embedder = QueryEmbedder()

    initial_state = {
        "query": query,
        "kanun_adi": kanun_adi,
        "top_k": 5,
        "embedder": embedder,
        "results": [],
        "answer": "",
        "is_grounded": False,
        "grading_reason": "",
        "retry_count": 0,
        "max_retries": 2,
    }

    try:
        final_state = compiled_graph.invoke(initial_state)
    except Exception as e:
        print(f"HATA: Soru işlenirken bir sorun oluştu: {e}")
        sys.exit(1)

    print("=== CEVAP ===")
    print(final_state["answer"])
    print()
    print(
        f"(grounded={final_state['is_grounded']}, "
        f"retry_count={final_state['retry_count']}, "
        f"grading_reason={final_state['grading_reason'] or '-'})"
    )
    print()

    print("=== KAYNAKLAR ===")
    results = final_state["results"]
    if not results:
        print("(ilgili bir madde bulunamadı)")
    else:
        for i, r in enumerate(dedupe_by_madde(results), 1):
            print(f"[{i}] {r.context_path} (mesafe: {r.distance:.4f})")


if __name__ == "__main__":
    main()
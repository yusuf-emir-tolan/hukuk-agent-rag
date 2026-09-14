from unittest.mock import patch
import graph as g


def _initial_state(query="test soru", results=None):
    return {
        "query": query,
        "kanun_adi": None,
        "top_k": 5,
        "embedder": None,
        "results": results or [],
        "answer": "",
        "is_grounded": False,
        "grading_reason": "",
        "retry_count": 0,
        "max_retries": 2,
    }


class TestDecideAfterRetrieve:
    def test_sonuc_yoksa_no_context(self):
        assert g.decide_after_retrieve({"results": []}) == "no_context"

    def test_dusuk_mesafeli_sonuc_generate_a_gider(self, make_chunk):
        results = [make_chunk("5237 TCK", "81", distance=0.30)]
        assert g.decide_after_retrieve({"results": results}) == "generate"

    def test_yuksek_mesafeli_sonuc_no_context_a_gider(self, make_chunk):
        results = [make_chunk("6098 TBK", "104", distance=0.45)]
        assert g.decide_after_retrieve({"results": results}) == "no_context"

    def test_en_iyi_sonuc_esigin_altindaysa_generate_a_gider(self, make_chunk):
        results = [
            make_chunk("6098 TBK", "104", distance=0.50),
            make_chunk("5237 TCK", "81", distance=0.35),
        ]
        assert g.decide_after_retrieve({"results": results}) == "generate"


class TestDecideAfterGrading:
    def test_destekliyse_bitir(self):
        state = {"is_grounded": True, "retry_count": 1, "max_retries": 2}
        assert g.decide_after_grading(state) == "end"

    def test_desteksiz_ve_deneme_hakki_varsa_tekrar_dene(self):
        state = {"is_grounded": False, "retry_count": 1, "max_retries": 2}
        assert g.decide_after_grading(state) == "retry"

    def test_desteksiz_ve_deneme_hakki_bittiyse_fallback(self):
        state = {"is_grounded": False, "retry_count": 2, "max_retries": 2}
        assert g.decide_after_grading(state) == "fallback"


class TestFullGraphFlow:

    def test_ilk_seferde_destekli_cevap(self, make_chunk):
        chunks = [make_chunk("5237 TCK", "81", distance=0.3)]
        calls = {"generate": 0, "grade": 0}

        with patch("graph.retrieve", return_value=chunks), \
             patch("graph.generate_answer") as mock_gen, \
             patch("graph.check_groundedness") as mock_grade:

            def fake_gen(query, chunks):
                calls["generate"] += 1
                return "CEVAP"

            def fake_grade(query, chunks, answer):
                calls["grade"] += 1
                return True, "destekli"

            mock_gen.side_effect = fake_gen
            mock_grade.side_effect = fake_grade

            final = g.build_graph().invoke(_initial_state())

            assert final["answer"] == "CEVAP"
            assert calls["generate"] == 1
            assert calls["grade"] == 1

    def test_desteksiz_cevap_duzeltilip_kabul_edilir(self, make_chunk):
        chunks = [make_chunk("5237 TCK", "81", distance=0.3)]

        def fake_gen(query, chunks):
            return "DUZELTILMIS" if "SİSTEM NOTU" in query else "HALUSINE"

        def fake_grade(query, chunks, answer):
            return (answer == "DUZELTILMIS"), "kontrol"

        with patch("graph.retrieve", return_value=chunks), \
             patch("graph.generate_answer", side_effect=fake_gen), \
             patch("graph.check_groundedness", side_effect=fake_grade):

            final = g.build_graph().invoke(_initial_state())
            assert final["answer"] == "DUZELTILMIS"

    def test_hep_desteksizse_fallback_mesaji_doner_ve_sonlanir(self, make_chunk):
        chunks = [make_chunk("5237 TCK", "81", distance=0.3)]
        calls = {"generate": 0}

        def fake_gen(query, chunks):
            calls["generate"] += 1
            return "HEP-HALUSINE"

        with patch("graph.retrieve", return_value=chunks), \
             patch("graph.generate_answer", side_effect=fake_gen), \
             patch("graph.check_groundedness", return_value=(False, "hiçbir zaman destekli değil")):

            final = g.build_graph().invoke(_initial_state())

            assert "hukuk uzmanına danışın" in final["answer"]
            assert calls["generate"] == 2

    def test_sonuc_yoksa_generate_hic_cagrilmaz(self):
        calls = {"generate": 0}

        def fake_gen(query, chunks):
            calls["generate"] += 1
            return "BU HİÇ ÇAĞRILMAMALI"

        with patch("graph.retrieve", return_value=[]), \
             patch("graph.generate_answer", side_effect=fake_gen):

            final = g.build_graph().invoke(_initial_state())

            assert calls["generate"] == 0
            assert "bulamadım" in final["answer"]

    def test_alakasiz_sonuclarda_generate_hic_cagrilmaz(self, make_chunk):
        irrelevant_chunks = [make_chunk("6098 TBK", "104", distance=0.50)]
        calls = {"generate": 0}

        def fake_gen(query, chunks):
            calls["generate"] += 1
            return "BU HİÇ ÇAĞRILMAMALI"

        with patch("graph.retrieve", return_value=irrelevant_chunks), \
             patch("graph.generate_answer", side_effect=fake_gen):

            final = g.build_graph().invoke(_initial_state())

            assert calls["generate"] == 0
            assert "bulamadım" in final["answer"]
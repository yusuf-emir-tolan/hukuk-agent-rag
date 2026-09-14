from madde_chunker import TurkishLegislationChunker


def make_chunker():
    return TurkishLegislationChunker()


class TestBasicMadde:
    def test_fikrasiz_madde_tek_chunk_uretir(self):
        c = make_chunker()
        text = "MADDE 1- Bu kanunun amacı budur."
        chunks = c.process_legislation(text, kanun_adi="TEST")

        assert len(chunks) == 1
        meta = chunks[0]["metadata"]
        assert meta["madde_no"] == "1"
        assert meta["fikra_no"] == "1"
        assert meta["bent_no"] is None
        assert meta["context_path"] == "TEST > MADDE 1"
        assert "amacı budur" in chunks[0]["page_content"]

    def test_birden_fazla_madde_ayri_ayri_ayristirilir(self):
        c = make_chunker()
        text = (
            "MADDE 1- Birinci madde metni.\n\n"
            "MADDE 2- İkinci madde metni.\n"
        )
        chunks = c.process_legislation(text, kanun_adi="TEST")
        madde_nos = [ch["metadata"]["madde_no"] for ch in chunks]
        assert madde_nos == ["1", "2"]

    def test_gecici_ve_ek_madde_tipleri_taninir(self):
        c = make_chunker()
        text = "GEÇİCİ MADDE 1- Geçiş hükmü metni."
        chunks = c.process_legislation(text, kanun_adi="TEST")
        assert len(chunks) == 1
        assert "GEÇİCİ MADDE 1" in chunks[0]["metadata"]["context_path"]


class TestFikraSplitting:
    def test_fikralar_ayri_chunklara_bolunur(self):
        c = make_chunker()
        text = (
            "MADDE 5- Bu maddenin açıklama cümlesi.\n"
            "(1) Birinci fıkra metni burada.\n"
            "(2) İkinci fıkra metni burada.\n"
        )
        chunks = c.process_legislation(text, kanun_adi="TEST")

        assert len(chunks) == 2
        assert [ch["metadata"]["fikra_no"] for ch in chunks] == ["1", "2"]
        assert "Birinci fıkra" in chunks[0]["page_content"]
        assert "İkinci fıkra" in chunks[1]["page_content"]

    def test_parent_text_maddenin_tum_metnini_korur(self):
        c = make_chunker()
        text = (
            "MADDE 5- Bu maddenin açıklama cümlesi.\n"
            "(1) Birinci fıkra metni burada.\n"
        )
        chunks = c.process_legislation(text, kanun_adi="TEST")
        assert "açıklama cümlesi" in chunks[0]["metadata"]["parent_text"]


class TestBentSplitting:
    def test_bentler_fikra_girisini_de_icerir(self):
        c = make_chunker()
        text = (
            "MADDE 10- Giriş cümlesi.\n"
            "(1) Fıkra girişi.\n"
            "a) Bent a metni.\n"
            "b) Bent b metni.\n"
        )
        chunks = c.process_legislation(text, kanun_adi="TEST")

        assert len(chunks) == 2
        bent_nos = [ch["metadata"]["bent_no"] for ch in chunks]
        assert bent_nos == ["a", "b"]

        for ch in chunks:
            assert "Fıkra girişi" in ch["page_content"]

        assert "Bent a metni" in chunks[0]["page_content"]
        assert "Bent b metni" in chunks[1]["page_content"]
        assert chunks[0]["metadata"]["context_path"] == "TEST > MADDE 10"


class TestHierarchy:
    def test_kitap_kisim_context_path_a_yansir(self):
        c = make_chunker()
        text = (
            "BİRİNCİ KİTAP: Genel Hükümler\n\n"
            "BİRİNCİ KISIM: Temel İlkeler\n\n"
            "MADDE 1- Birinci madde metni.\n\n"
            "İKİNCİ KISIM: İkinci Kısım Başlığı\n\n"
            "MADDE 2- İkinci madde metni.\n"
        )
        chunks = c.process_legislation(text, kanun_adi="TEST")

        assert "KİTAP BİRİNCİ" in chunks[0]["metadata"]["context_path"]
        assert "KISIM BİRİNCİ" in chunks[0]["metadata"]["context_path"]
        assert "KISIM İKİNCİ" in chunks[1]["metadata"]["context_path"]
        assert "KİTAP BİRİNCİ" in chunks[1]["metadata"]["context_path"]

    def test_yeni_kitap_alt_hiyerarsiyi_sifirlar(self):
        c = make_chunker()
        text = (
            "BİRİNCİ KİTAP: İlk Kitap\n\n"
            "BİRİNCİ KISIM: İlk Kısım\n\n"
            "MADDE 1- Metin.\n\n"
            "İKİNCİ KİTAP: İkinci Kitap\n\n"
            "MADDE 2- Metin.\n"
        )
        chunks = c.process_legislation(text, kanun_adi="TEST")
        assert "KİTAP İKİNCİ" in chunks[1]["metadata"]["context_path"]
        assert "KISIM BİRİNCİ" not in chunks[1]["metadata"]["context_path"]


class TestClassifyBaslik:
    """Tek harfli başlıkların harf mi yoksa Romen rakamı mı olduğunu
    ayırt eden mantık - chunker'ın en kırılgan parçası."""

    def test_ilk_baslik_a_beklenir(self):
        c = make_chunker()
        assert c._classify_baslik("A", None) is True

    def test_sirali_harf_dogru_taninir(self):
        c = make_chunker()
        assert c._classify_baslik("B", "A") is True

    def test_sira_disi_tek_harf_romen_sayilir(self):
        c = make_chunker()
        assert c._classify_baslik("I", "A") is False

    def test_ilk_baslik_a_disinda_bir_sey_ise_romen_sayilir(self):
        c = make_chunker()
        assert c._classify_baslik("V", None) is False

    def test_coklu_karakter_tamamen_romen_degilse_harf_sayilir(self):
        c = make_chunker()
        assert c._classify_baslik("AB", None) is True

    def test_coklu_karakter_tamamen_romen_ise_romen_sayilir(self):
        c = make_chunker()
        assert c._classify_baslik("II", None) is False

    def test_tartismasiz_harf_dogrudan_harf_sayilir(self):
        c = make_chunker()
        assert c._classify_baslik("F", "A") is True


class TestCleanText:
    def test_dipnot_gibi_gorunen_paragraflar_temizlenir(self):
        c = make_chunker()
        text = (
            "MADDE 1- Asıl madde metni.\n\n"
            "5378 sayılı Kanunun 1/1/2020 tarihli değişikliği ile eklenmiştir.\n\n"
            "MADDE 2- İkinci madde metni.\n"
        )
        cleaned = c.clean_text(text)
        assert "tarihli" not in cleaned.lower() or "sayılı" not in cleaned.lower()

    def test_bos_metin_bos_dondurur(self):
        c = make_chunker()
        assert c.clean_text("") == ""

    def test_sayfa_numaralari_temizlenir(self):
        c = make_chunker()
        text = "MADDE 1- Metin burada.\n\n42\n\nMADDE 2- Devam metni."
        cleaned = c.clean_text(text)
        assert "\n42\n" not in cleaned


class TestEmptyAndEdgeCases:
    def test_hic_madde_yoksa_bos_liste_doner(self):
        c = make_chunker()
        chunks = c.process_legislation("Sadece düz metin, hiç madde yok.", kanun_adi="TEST")
        assert chunks == []

    def test_uzun_fikra_metni_fallback_split_ile_bolunur(self):
        c = make_chunker()
        uzun_cumle = "Bu çok uzun bir fıkra metnidir. " * 100
        text = f"MADDE 1- Giriş.\n(1) {uzun_cumle}\n"
        chunks = c.process_legislation(text, kanun_adi="TEST", max_fikra_chars=200)
        assert len(chunks) > 1
        assert all(ch["metadata"]["madde_no"] == "1" for ch in chunks)
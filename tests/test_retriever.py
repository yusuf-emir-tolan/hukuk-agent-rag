from unittest.mock import MagicMock, patch
import psycopg2
import pytest

import retriever as r


class TestDedupeByMadde:
    def test_ayni_madde_birden_fazla_gelirse_tekillestirir(self, make_chunk):
        chunks = [
            make_chunk("5237 TCK", "82", distance=0.28),
            make_chunk("5237 TCK", "81", distance=0.32),
            make_chunk("5237 TCK", "82", fikra_no="3", distance=0.34),  # tekrar
        ]
        result = r.dedupe_by_madde(chunks)
        assert [c.madde_no for c in result] == ["82", "81"]

    def test_en_alakali_olan_ilk_gorulen_tutulur(self, make_chunk):
        chunks = [
            make_chunk("4721 TMK", "8", fikra_no="1", distance=0.20),
            make_chunk("4721 TMK", "8", fikra_no="2", distance=0.50),
        ]
        result = r.dedupe_by_madde(chunks)
        assert len(result) == 1
        assert result[0].fikra_no == "1"

    def test_bos_liste(self):
        assert r.dedupe_by_madde([]) == []

    def test_farkli_kanunlarda_ayni_madde_no_farkli_sayilir(self, make_chunk):
        chunks = [
            make_chunk("5237 TCK", "81", distance=0.3),
            make_chunk("4721 TMK", "81", distance=0.4),
        ]
        result = r.dedupe_by_madde(chunks)
        assert len(result) == 2


class TestConnectionPool:
    def setup_method(self):
        r._pool = None
        r._pool_db_url = None

    def test_get_pool_ayni_url_icin_ayni_havuzu_dondurur(self):
        with patch("retriever.pg_pool.ThreadedConnectionPool") as MockPoolClass:
            MockPoolClass.return_value = MagicMock()
            p1 = r.get_pool("postgresql://fake")
            p2 = r.get_pool("postgresql://fake")
            assert MockPoolClass.call_count == 1
            assert p1 is p2

    def test_close_pool_sonrasi_tekrar_olusturur(self):
        with patch("retriever.pg_pool.ThreadedConnectionPool") as MockPoolClass:
            MockPoolClass.return_value = MagicMock()
            r.get_pool("postgresql://fake")
            r.close_pool()
            r.get_pool("postgresql://fake")
            assert MockPoolClass.call_count == 2


class TestSearchRetry:
    def setup_method(self):
        r._pool = None
        r._pool_db_url = None

    def test_basarili_sorguda_baglanti_geri_verilir(self):
        with patch("retriever.pg_pool.ThreadedConnectionPool") as MockPoolClass:
            mock_pool = MagicMock()
            MockPoolClass.return_value = mock_pool

            fake_conn = MagicMock()
            fake_cursor = MagicMock()
            fake_cursor.fetchall.return_value = [
                ("5237 TCK", "81", "1", None, "yol", "metin", 0.3)
            ]
            fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
            mock_pool.getconn.return_value = fake_conn

            results = r.search("postgresql://fake", [0.1, 0.2], top_k=5)

            assert len(results) == 1
            mock_pool.putconn.assert_called_once_with(fake_conn)

    def test_operational_error_retry_eder_ve_bozuk_baglantiyi_atar(self):
        with patch("retriever.pg_pool.ThreadedConnectionPool") as MockPoolClass:
            mock_pool = MagicMock()
            MockPoolClass.return_value = mock_pool

            call_count = {"n": 0}

            def fake_getconn():
                call_count["n"] += 1
                conn = MagicMock()
                if call_count["n"] == 1:
                    conn.cursor.return_value.__enter__.side_effect = (
                        psycopg2.OperationalError("bağlantı koptu")
                    )
                else:
                    conn.cursor.return_value.__enter__.return_value.fetchall.return_value = []
                return conn

            mock_pool.getconn.side_effect = fake_getconn

            results = r.search("postgresql://fake", [0.1, 0.2], top_k=5, max_retries=2)

            assert results == []
            assert call_count["n"] == 2
            first_putconn_call = mock_pool.putconn.call_args_list[0]
            assert first_putconn_call.kwargs.get("close") is True

    def test_tum_denemeler_basarisizsa_hata_firlatir(self):
        with patch("retriever.pg_pool.ThreadedConnectionPool") as MockPoolClass:
            mock_pool = MagicMock()
            MockPoolClass.return_value = mock_pool

            def always_fail():
                conn = MagicMock()
                conn.cursor.return_value.__enter__.side_effect = (
                    psycopg2.OperationalError("hep başarısız")
                )
                return conn

            mock_pool.getconn.side_effect = always_fail

            with pytest.raises(psycopg2.OperationalError):
                r.search("postgresql://fake", [0.1, 0.2], top_k=5, max_retries=2)


class TestWaitForPool:
    def setup_method(self):
        r._pool = None
        r._pool_db_url = None

    def test_gecici_basarisizliktan_sonra_basarili_olur(self):
        call_count = {"n": 0}

        def fake_ctor(minconn, maxconn, db_url):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise psycopg2.OperationalError("henüz hazır değil")
            return MagicMock()

        with patch("retriever.pg_pool.ThreadedConnectionPool", side_effect=fake_ctor), \
             patch("retriever.time.sleep"):
            r.wait_for_pool("postgresql://fake", max_attempts=5, delay=0.01)
            assert call_count["n"] == 3

    def test_hicbir_zaman_basarili_olmazsa_hata_firlatir(self):
        with patch(
            "retriever.pg_pool.ThreadedConnectionPool",
            side_effect=psycopg2.OperationalError("hiç hazır olmuyor"),
        ), patch("retriever.time.sleep"):
            with pytest.raises(RuntimeError):
                r.wait_for_pool("postgresql://fake", max_attempts=3, delay=0.01)
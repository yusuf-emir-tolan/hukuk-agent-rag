import re
from typing import Any, Dict, List, Optional


class TurkishLegislationChunker:

    def __init__(self):
        _ORDINAL_WORDS = (
            "BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|"
            "DOKUZUNCU|ONUNCU|ON BİRİNCİ|ON İKİNCİ|ON ÜÇÜNCÜ|ON DÖRDÜNCÜ|"
            "ON BEŞİNCİ|ON ALTINCI|ON YEDİNCİ|ON SEKİZİNCİ|ON DOKUZUNCU|"
            "YİRMİNCİ|YİRMİ BİRİNCİ|YİRMİ İKİNCİ|YİRMİ ÜÇÜNCÜ|YİRMİ DÖRDÜNCÜ|"
            "YİRMİ BEŞİNCİ|YİRMİ ALTINCI|YİRMİ YEDİNCİ|YİRMİ SEKİZİNCİ|"
            "YİRMİ DOKUZUNCU|OTUZUNCU"
        )
        self.HIYERARSI_PATTERN = re.compile(
            r"^(?:"
            r"(?P<no_a>" + _ORDINAL_WORDS + r")\s+(?P<tip_a>KİTAP|KISIM|BÖLÜM|CİLT)"
            r"|"
            r"(?P<tip_b>KİTAP|KISIM|BÖLÜM|CİLT)\s+(?P<no_b>[0-9IVXLCDM]+|[A-ZÇĞİÖŞÜ]+)"
            r")\b"
            r"(?:\s*[\:\-\–\—\.]?[ \t]*(?P<baslik>[^\r\n]+))?",
            re.MULTILINE,
        )

        self.MADDE_PATTERN = re.compile(
            r"^(?P<madde_tipi>(?:GEÇİCİ\s+|EK\s+)?MADDE)\s+(?P<no>\d+[A-Z\d]*)\s*[\-\–\—\.\:]?",
            re.MULTILINE | re.IGNORECASE,
        )

        self.BASLIK_ADAY_PATTERN = re.compile(
            r"^(?P<no>[A-ZÇĞİÖŞÜ]{1,4})\.\s+(?P<baslik>[^\r\n]+)", re.MULTILINE
        )
        self._ROMEN_CHARS = set("IVXLCDM")
        self._HARF_SIRA = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"

        self.FIKRA_PATTERN = re.compile(
            r"^\((?P<no>\d+[a-z]?)\)\s*", re.MULTILINE
        )

        self.BENT_PATTERN = re.compile(
            r"^(?P<harf>[a-zçğıöşü]\d?)\)\s*", re.MULTILINE
        )

        self._TITLE_MAX_LEN = 200

        self._FOOTNOTE_KEYWORDS = ("tarihli", "sayılı")
        self._PARA_SPLIT = re.compile(r"\n{2,}")

        self.SAYFA_NO_PATTERN = re.compile(r"\n[ \t]*\d{1,4}[ \t]*\n")

    def _classify_baslik(self, no: str, last_harf: Optional[str]) -> bool:


        if len(no) > 1:
            return not all(ch in self._ROMEN_CHARS for ch in no)

        if no not in self._ROMEN_CHARS:
            return True  # tartışmasız harf (B, E, F, G, ... gibi - roma rakamı olamaz)

        expected_next = (
            "A"
            if last_harf is None
            else self._HARF_SIRA[self._HARF_SIRA.find(last_harf) + 1]
            if last_harf in self._HARF_SIRA
            and self._HARF_SIRA.find(last_harf) + 1 < len(self._HARF_SIRA)
            else None
        )
        return no == expected_next

    def _strip_footnote_paragraphs(self, text: str) -> str:
        paragraphs = self._PARA_SPLIT.split(text)
        kept = []
        for para in paragraphs:
            stripped = para.strip()
            low = stripped.lower()
            looks_like_footnote = (
                stripped
                and len(stripped) < 500
                and all(kw in low for kw in self._FOOTNOTE_KEYWORDS)
                and not self.MADDE_PATTERN.search(stripped)
                and not self.HIYERARSI_PATTERN.search(stripped)
            )
            if looks_like_footnote:
                continue
            kept.append(para)
        return "\n\n".join(kept)

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\x0c", "\n\n")

        text = re.sub(r"(?m)^[ \t]+", "", text)

        text = self._strip_footnote_paragraphs(text)

        text = self.SAYFA_NO_PATTERN.sub("\n\n", text)

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]*\n+", "\n\n", text)

        return text.strip()

    def _fallback_split(
        self, text: str, max_chars: int = 1500, overlap: int = 200
    ) -> List[str]:
        if len(text) <= max_chars:
            return [text]

        chunks = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                overlap_text = (
                    current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                )
                current_chunk = overlap_text + (" " if overlap_text else "") + sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _find_madde_title_boundary(
        self, text: str, madde_pos: int, min_pos: int
    ) -> int:

        j = madde_pos

        if j > min_pos and text[j - 1] == "\n":
            j -= 1
        else:
            return madde_pos

        line_end = j
        line_start = text.rfind("\n", min_pos, line_end)
        line_start = line_start + 1 if line_start != -1 else min_pos

        title_line = text[line_start:line_end].strip()

        if (
            title_line
            and len(title_line) <= self._TITLE_MAX_LEN
            and not self.MADDE_PATTERN.match(title_line)
            and not self.FIKRA_PATTERN.match(title_line)
            and not self.HIYERARSI_PATTERN.match(title_line)
        ):
            return line_start

        return madde_pos

    def process_legislation(
        self,
        full_text: str,
        kanun_adi: str = "MEVZUAT",
        max_fikra_chars: int = 1500,
    ) -> List[Dict[str, Any]]:
        cleaned_text = self.clean_text(full_text)

        structural_matches = []

        for m in self.HIYERARSI_PATTERN.finditer(cleaned_text):
            tip = m.group("tip_a") or m.group("tip_b")
            no = m.group("no_a") or m.group("no_b")
            structural_matches.append(
                {
                    "type": "HIYERARSI",
                    "pos": m.start(),
                    "match": m,
                    "tip": tip.upper(),
                    "no": no,
                    "baslik": (m.group("baslik") or "").strip(),
                }
            )

        for m in self.BASLIK_ADAY_PATTERN.finditer(cleaned_text):
            structural_matches.append(
                {
                    "type": "BASLIK_ADAY",
                    "pos": m.start(),
                    "match": m,
                    "no": m.group("no"),
                    "baslik": m.group("baslik").strip(),
                }
            )

        for m in self.MADDE_PATTERN.finditer(cleaned_text):
            structural_matches.append(
                {
                    "type": "MADDE",
                    "pos": m.start(),
                    "match": m,
                    "madde_tipi": m.group("madde_tipi"),
                    "no": m.group("no"),
                }
            )

        structural_matches.sort(key=lambda x: x["pos"])

        for idx, item in enumerate(structural_matches):
            if item["type"] != "MADDE":
                continue
            min_pos = structural_matches[idx - 1]["pos"] if idx > 0 else 0
            item["boundary_pos"] = self._find_madde_title_boundary(
                cleaned_text, item["pos"], min_pos
            )

        chunks = []
        active_hierarchy = {
            "KİTAP": None,
            "KISIM": None,
            "BÖLÜM": None,
            "CİLT": None,
            "HARF_BASLIK": None,
            "ROMEN_BASLIK": None,
        }
        last_harf = None

        for i, item in enumerate(structural_matches):
            if item["type"] == "HIYERARSI":
                tip = item["tip"]
                baslik_str = f"{tip} {item['no']}"
                if item["baslik"]:
                    baslik_str += f": {item['baslik']}"
                active_hierarchy[tip] = baslik_str

                # Alt hiyerarşileri sıfırla
                if tip == "KİTAP":
                    active_hierarchy["KISIM"] = None
                    active_hierarchy["BÖLÜM"] = None
                elif tip == "KISIM":
                    active_hierarchy["BÖLÜM"] = None
                active_hierarchy["HARF_BASLIK"] = None
                active_hierarchy["ROMEN_BASLIK"] = None
                last_harf = None

            elif item["type"] == "BASLIK_ADAY":
                no = item["no"]
                is_harf = self._classify_baslik(no, last_harf)
                if is_harf:
                    active_hierarchy["HARF_BASLIK"] = f"{no}. {item['baslik']}"
                    active_hierarchy["ROMEN_BASLIK"] = None
                    last_harf = no
                else:
                    active_hierarchy["ROMEN_BASLIK"] = f"{no}. {item['baslik']}"

            elif item["type"] == "MADDE":
                start_pos = item["match"].end()

                next_pos = len(cleaned_text)
                if i + 1 < len(structural_matches):
                    nxt = structural_matches[i + 1]
                    next_pos = nxt.get("boundary_pos", nxt["pos"])

                madde_raw_text = cleaned_text[start_pos:next_pos].strip()
                madde_no = f"{item['madde_tipi']} {item['no']}"

                path_components = [kanun_adi]
                for key in [
                    "KİTAP",
                    "KISIM",
                    "BÖLÜM",
                    "CİLT",
                    "HARF_BASLIK",
                    "ROMEN_BASLIK",
                ]:
                    if active_hierarchy[key]:
                        path_components.append(active_hierarchy[key])
                path_components.append(madde_no)

                context_path = " > ".join(path_components)

                parent_text = f"[{context_path}]\n{madde_raw_text}"

                fikra_matches = list(self.FIKRA_PATTERN.finditer(madde_raw_text))

                if not fikra_matches:
                    sub_chunks = self._fallback_split(
                        madde_raw_text, max_chars=max_fikra_chars
                    )
                    for idx2, chunk_text in enumerate(sub_chunks):
                        chunks.append(
                            {
                                "page_content": f"[{context_path}]\n{chunk_text}",
                                "metadata": {
                                    "kanun_adi": kanun_adi,
                                    "madde_no": item["no"],
                                    "madde_tipi": item["madde_tipi"],
                                    "fikra_no": "1" if len(sub_chunks) == 1 else f"1.{idx2+1}",
                                    "bent_no": None,
                                    "context_path": context_path,
                                    "parent_text": parent_text,
                                },
                            }
                        )
                else:
                    for f_idx, f_match in enumerate(fikra_matches):
                        f_start = f_match.end()
                        f_next = (
                            fikra_matches[f_idx + 1].start()
                            if f_idx + 1 < len(fikra_matches)
                            else len(madde_raw_text)
                        )

                        fikra_no = f_match.group("no")
                        fikra_text = madde_raw_text[f_start:f_next].strip()

                        bent_matches = list(self.BENT_PATTERN.finditer(fikra_text))

                        if not bent_matches:
                            sub_chunks = self._fallback_split(
                                fikra_text, max_chars=max_fikra_chars
                            )
                            for idx2, chunk_text in enumerate(sub_chunks):
                                f_no_str = (
                                    fikra_no
                                    if len(sub_chunks) == 1
                                    else f"{fikra_no}.{idx2+1}"
                                )
                                chunks.append(
                                    {
                                        "page_content": f"[{context_path} > Fıkra {f_no_str}]\n({fikra_no}) {chunk_text}",
                                        "metadata": {
                                            "kanun_adi": kanun_adi,
                                            "madde_no": item["no"],
                                            "madde_tipi": item["madde_tipi"],
                                            "fikra_no": f_no_str,
                                            "bent_no": None,
                                            "context_path": context_path,
                                            "parent_text": parent_text,
                                        },
                                    }
                                )
                        else:
                            first_bent_start = bent_matches[0].start()
                            main_fikra_intro = fikra_text[:first_bent_start].strip()

                            for b_idx, b_match in enumerate(bent_matches):
                                b_start = b_match.end()
                                b_next = (
                                    bent_matches[b_idx + 1].start()
                                    if b_idx + 1 < len(bent_matches)
                                    else len(fikra_text)
                                )

                                bent_harf = b_match.group("harf")
                                bent_body = fikra_text[b_start:b_next].strip()

                                combined_bent_text = (
                                    f"({fikra_no}) {main_fikra_intro}\n{bent_harf}) {bent_body}"
                                    if main_fikra_intro
                                    else f"({fikra_no}) {bent_harf}) {bent_body}"
                                )

                                chunks.append(
                                    {
                                        "page_content": f"[{context_path} > Fıkra {fikra_no} > Bent {bent_harf}]\n{combined_bent_text}",
                                        "metadata": {
                                            "kanun_adi": kanun_adi,
                                            "madde_no": item["no"],
                                            "madde_tipi": item["madde_tipi"],
                                            "fikra_no": fikra_no,
                                            "bent_no": bent_harf,
                                            "context_path": context_path,
                                            "parent_text": parent_text,
                                        },
                                    }
                                )

        return chunks
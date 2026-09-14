import os
import sys
import types

os.environ.setdefault("GROQ_API_KEY", "test-fake-key")

import pytest
from dataclasses import dataclass
from typing import Optional

@dataclass
class FakeChunk:

    kanun_adi: str
    madde_no: str
    fikra_no: Optional[str]
    bent_no: Optional[str]
    context_path: str
    parent_text: str
    distance: float

@pytest.fixture
def make_chunk():


    def _make(kanun_adi, madde_no, fikra_no="1", bent_no=None,
              context_path=None, parent_text="test metni", distance=0.3):
        return FakeChunk(
            kanun_adi=kanun_adi,
            madde_no=madde_no,
            fikra_no=fikra_no,
            bent_no=bent_no,
            context_path=context_path or f"{kanun_adi} > Madde {madde_no}",
            parent_text=parent_text,
            distance=distance,
        )

    return _make
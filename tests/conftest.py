import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workbench.data import generate_dataset  # noqa: E402

AS_OF = pd.Timestamp("2026-11-03")


@pytest.fixture(scope="session")
def ds():
    return generate_dataset(seed=42, as_of=AS_OF)


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Tests never call the live model."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

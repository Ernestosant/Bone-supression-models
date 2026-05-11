from __future__ import annotations

import pytest

from bone_suppression.model_io import load_model
from bone_suppression.registry import get_model_spec


def test_load_model_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        load_model(get_model_spec("gan_mso2"), "missing.keras", device="gpu")

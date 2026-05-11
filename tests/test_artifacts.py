from __future__ import annotations

from bone_suppression.artifacts import sha256_file


def test_sha256_file(tmp_path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("bone suppression\n", encoding="utf-8")

    expected = "b103e913081630762a48490dc5cfcbcf833b1fc6e2a21e4cea4ccf5212295913"

    assert sha256_file(payload) == expected

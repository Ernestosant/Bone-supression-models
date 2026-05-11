from __future__ import annotations

from bone_suppression.artifacts import sha256_file


def test_sha256_file(tmp_path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_bytes(b"bone suppression\n")

    expected = "b574388b49306ba6bf7661de4f39974a1ede7c4a7e26262d18e93b87f91bd465"

    assert sha256_file(payload) == expected

"""Tests for Block serialization."""
import pytest
import json
from chatboard.block.block12 import Block


class TestBlockSerialization:
    """Tests for Block serialization."""

    def test_model_dump(self):
        with Block("Root", role="user", tags=["msg"]) as root:
            root /= Block("Hello", tags=["greeting"])

        data = root.model_dump()

        assert isinstance(data, dict)
        assert "role" in data or "_text" in data

    def test_model_load(self):
        with Block("Root", role="user", tags=["msg"]) as root:
            root /= Block("Hello", tags=["greeting"])

        data = root.model_dump()
        restored = Block.model_load(data)

        assert restored.text == root.text

    def test_serialization_roundtrip(self):
        with Block("Root", role="user", tags=["msg"]) as root:
            root /= Block("Hello", tags=["greeting"])

        data = root.model_dump()
        restored = Block.model_load(data)

        assert restored.text == root.text
        assert restored.role == root.role


class TestBlockJsonCompatibility:
    """Tests for JSON compatibility."""

    def test_json_serializable(self):
        with Block("Test") as block:
            block /= "Child"

        data = block.model_dump()

        json_str = json.dumps(data)
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

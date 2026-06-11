from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

import server


class EnvironmentTests(unittest.TestCase):
    def test_required_env_rejects_missing_and_empty_values(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "REQUIRED_PATH"):
                server.required_env("REQUIRED_PATH")

        with patch.dict(os.environ, {"REQUIRED_PATH": "  "}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "REQUIRED_PATH"):
                server.required_env("REQUIRED_PATH")

    def test_required_env_returns_configured_value(self) -> None:
        with patch.dict(os.environ, {"REQUIRED_PATH": "~/models"}, clear=True):
            self.assertEqual(server.required_env("REQUIRED_PATH"), "~/models")


def write_gguf(path: Path, fields: list[tuple[str, int, object]]) -> None:
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<I", 3))
        handle.write(struct.pack("<Q", 0))
        handle.write(struct.pack("<Q", len(fields)))
        for key, value_type, value in fields:
            key_bytes = key.encode()
            handle.write(struct.pack("<Q", len(key_bytes)))
            handle.write(key_bytes)
            handle.write(struct.pack("<I", value_type))
            if value_type == 8:
                value_bytes = str(value).encode()
                handle.write(struct.pack("<Q", len(value_bytes)))
                handle.write(value_bytes)
            elif value_type == 4:
                handle.write(struct.pack("<I", int(value)))
            else:
                raise AssertionError(f"unsupported test value type: {value_type}")


class GgufMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        server.GGUF_METADATA_CACHE.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_embedding_metadata(self) -> None:
        model = self.root / "embedding.gguf"
        write_gguf(
            model,
            [
                ("general.architecture", 8, "gemma-embedding"),
                ("gemma-embedding.embedding_length", 4, 5376),
                ("gemma-embedding.pooling_type", 4, 3),
            ],
        )

        metadata = server.read_gguf_metadata(model)

        self.assertEqual(metadata["architecture"], "gemma-embedding")
        self.assertEqual(metadata["pooling"], "last")
        self.assertEqual(metadata["embedding_dimensions"], 5376)
        self.assertEqual(metadata["detected_mode"], "embeddings")

    def test_pooling_modes_and_corrupt_file(self) -> None:
        for value, name, mode in [
            (1, "mean", "embeddings"),
            (2, "cls", "embeddings"),
            (3, "last", "embeddings"),
            (4, "rank", "rerank"),
        ]:
            with self.subTest(pooling=name):
                model = self.root / f"{name}.gguf"
                write_gguf(
                    model,
                    [
                        ("general.architecture", 8, "bert"),
                        ("bert.pooling_type", 4, value),
                    ],
                )
                metadata = server.read_gguf_metadata(model)
                self.assertEqual(metadata["pooling"], name)
                self.assertEqual(metadata["detected_mode"], mode)

        corrupt = self.root / "corrupt.gguf"
        corrupt.write_bytes(b"not a gguf")
        metadata = server.read_gguf_metadata(corrupt)
        self.assertEqual(metadata["detected_mode"], "chat")
        self.assertIsNotNone(metadata["metadata_error"])

    def test_cache_invalidates_when_file_changes(self) -> None:
        model = self.root / "model.gguf"
        write_gguf(model, [("general.architecture", 8, "llama")])
        first = server.read_gguf_metadata(model)
        self.assertEqual(first["detected_mode"], "chat")

        write_gguf(
            model,
            [
                ("general.architecture", 8, "gemma-embedding"),
                ("gemma-embedding.pooling_type", 4, 3),
            ],
        )
        os.utime(model, None)
        second = server.read_gguf_metadata(model)
        self.assertEqual(second["detected_mode"], "embeddings")


class BackendSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        server.GGUF_METADATA_CACHE.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model = self.root / "embedding.gguf"
        write_gguf(
            self.model,
            [
                ("general.architecture", 8, "gemma-embedding"),
                ("gemma-embedding.pooling_type", 4, 3),
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_normalization_and_command(self) -> None:
        settings = server.normalize_backend_settings(
            "embedding.gguf",
            self.model,
            {"mode": "auto", "pooling": "auto"},
        )
        self.assertEqual(settings["effective_mode"], "embeddings")
        self.assertEqual(settings["effective_pooling"], "last")

        fake_bin_dir = self.root / "bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)
        with patch.object(server, "LLAMA_BIN_DIR", fake_bin_dir):
            command = server.build_llama_command(settings, model=self.model, port=9999)
        self.assertIn("--embeddings", command)
        self.assertEqual(command[command.index("--pooling") + 1], "last")

    def test_manual_embedding_requires_pooling_when_unknown(self) -> None:
        model = self.root / "unknown.gguf"
        write_gguf(model, [("general.architecture", 8, "llama")])
        with self.assertRaisesRegex(ValueError, "pooling must be set"):
            server.normalize_backend_settings(
                "unknown.gguf",
                model,
                {"mode": "embeddings", "pooling": "auto"},
            )


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_latest_backend_is_filtered_by_purpose(self) -> None:
        registry = server.BackendRegistry()

        class FakeInstance:
            def __init__(self, mode: str, started_at: float) -> None:
                self.effective_mode = mode
                self.started_at = started_at

            def is_active(self) -> bool:
                return True

        chat = FakeInstance("chat", 20)
        embedding = FakeInstance("embeddings", 10)
        registry.instances = {"chat": chat, "embedding": embedding}

        selected = await registry.latest_active_instance(purpose="embeddings")

        self.assertIs(selected, embedding)

    def test_capability_errors(self) -> None:
        response = server.model_capability_error("chat.gguf", "chat", "embeddings")
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.body)
        self.assertEqual(payload["error"]["code"], "model_not_embedding_capable")


class EmbeddingsApiTests(unittest.TestCase):
    def test_dimensions_is_rejected(self) -> None:
        with TestClient(server.app) as client:
            response = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 128})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "unsupported_parameter")

    def test_embedding_response_is_forwarded(self) -> None:
        instance = type("Instance", (), {"model_id": "embedding.gguf", "backend_url": "http://backend"})()
        backend_response = httpx.Response(
            200,
            json={
                "object": "list",
                "model": "embedding.gguf",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json, headers):
                self.url = url
                self.json = json
                return backend_response

        with (
            patch.object(
                server.registry,
                "backend_for_request",
                AsyncMock(return_value=(instance, None)),
            ),
            patch.object(server.httpx, "AsyncClient", return_value=FakeClient()),
            TestClient(server.app) as client,
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "embedding.gguf", "input": ["hello"], "encoding_format": "float"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["embedding"], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()

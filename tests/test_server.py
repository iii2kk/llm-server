from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

import server
from llm_server.request_logs import RequestResponseLogger


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
            elif value_type == 9:
                handle.write(struct.pack("<I", 8))
                handle.write(struct.pack("<Q", len(value)))
                for item in value:
                    value_bytes = str(item).encode()
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

    def test_mtp_metadata(self) -> None:
        model = self.root / "mtp.gguf"
        write_gguf(
            model,
            [
                ("general.architecture", 8, "qwen35"),
                ("qwen35.nextn_predict_layers", 4, 1),
            ],
        )

        metadata = server.read_gguf_metadata(model)

        self.assertTrue(metadata["mtp_supported"])
        self.assertEqual(metadata["mtp_layers"], 1)

    def test_mtp_metadata_after_tokenizer_tokens(self) -> None:
        model = self.root / "mtp-after-tokenizer.gguf"
        write_gguf(
            model,
            [
                ("general.architecture", 8, "qwen35"),
                ("tokenizer.ggml.tokens", 9, ["a", "b"]),
                ("qwen35.nextn_predict_layers", 4, 1),
            ],
        )

        metadata = server.read_gguf_metadata(model)

        self.assertTrue(metadata["mtp_supported"])
        self.assertEqual(metadata["mtp_layers"], 1)


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
        command = server.build_llama_command(
            settings,
            model=self.model,
            port=9999,
            llama_bin_dir=fake_bin_dir,
        )
        self.assertIn("--embeddings", command)
        self.assertEqual(command[command.index("--pooling") + 1], "last")
        self.assertNotIn("--direct-io", command)

    def test_rocm_command_enables_direct_io(self) -> None:
        fake_bin_dir = self.root / "rocm-bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        with patch.object(server, "LLAMA_BIN_DIRS", {"rocm": fake_bin_dir}):
            command = server.build_llama_command(
                {"backend": "rocm"},
                model=self.model,
                port=9999,
                llama_bin_dir=fake_bin_dir,
            )

        self.assertIn("--direct-io", command)

    def test_reasoning_budget_is_added_to_command(self) -> None:
        fake_bin_dir = self.root / "bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        command = server.build_llama_command(
            {"reasoning": "on", "reasoning_budget": 512},
            model=self.model,
            port=9999,
            llama_bin_dir=fake_bin_dir,
        )

        self.assertEqual(command[command.index("--reasoning-budget") + 1], "512")

    def test_reasoning_budget_rejects_values_below_minus_one(self) -> None:
        fake_bin_dir = self.root / "bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        with self.assertRaisesRegex(ValueError, "reasoning_budget"):
            server.build_llama_command(
                {"reasoning_budget": -2},
                model=self.model,
                port=9999,
                llama_bin_dir=fake_bin_dir,
            )

    def test_mtp_auto_detection_adds_speculative_flags(self) -> None:
        model = self.root / "mtp.gguf"
        write_gguf(
            model,
            [
                ("general.architecture", 8, "qwen35"),
                ("qwen35.nextn_predict_layers", 4, 1),
            ],
        )
        settings = server.normalize_backend_settings(
            "mtp.gguf",
            model,
            {"mtp": "auto", "mtp_draft_tokens": 4},
        )
        self.assertTrue(settings["effective_mtp"])

        fake_bin_dir = self.root / "mtp-bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        command = server.build_llama_command(
            settings,
            model=model,
            port=9999,
            llama_bin_dir=fake_bin_dir,
        )

        self.assertEqual(command[command.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "4")

    def test_mtp_off_does_not_add_speculative_flags(self) -> None:
        model = self.root / "mtp-off.gguf"
        write_gguf(
            model,
            [
                ("general.architecture", 8, "qwen35"),
                ("qwen35.nextn_predict_layers", 4, 1),
            ],
        )
        settings = server.normalize_backend_settings("mtp-off.gguf", model, {"mtp": "off"})
        self.assertFalse(settings["effective_mtp"])

        fake_bin_dir = self.root / "mtp-off-bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        command = server.build_llama_command(
            settings,
            model=model,
            port=9999,
            llama_bin_dir=fake_bin_dir,
        )

        self.assertNotIn("--spec-type", command)

    def test_mtp_on_requires_mtp_metadata(self) -> None:
        model = self.root / "chat.gguf"
        write_gguf(model, [("general.architecture", 8, "llama")])
        with self.assertRaisesRegex(ValueError, "no nextn_predict_layers"):
            server.normalize_backend_settings(
                "chat.gguf",
                model,
                {"mtp": "on"},
            )

    def test_mtp_draft_tokens_must_be_positive_integer(self) -> None:
        model = self.root / "mtp-invalid.gguf"
        write_gguf(
            model,
            [
                ("general.architecture", 8, "qwen35"),
                ("qwen35.nextn_predict_layers", 4, 1),
            ],
        )
        for value in (0, -1, 1.5, True, "x"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "mtp_draft_tokens"):
                    server.normalize_backend_settings(
                        "mtp-invalid.gguf",
                        model,
                        {"mtp_draft_tokens": value},
                    )

    def test_backend_selection_is_normalized_and_validated(self) -> None:
        with patch.object(
            server,
            "LLAMA_BIN_DIRS",
            {"vulkan": Path("/vulkan"), "rocm": Path("/rocm")},
        ):
            settings = server.normalize_backend_settings(
                "embedding.gguf",
                self.model,
                {"backend": "rocm", "mode": "auto", "pooling": "auto"},
            )
            self.assertEqual(settings["backend"], "rocm")

            with self.assertRaisesRegex(ValueError, "backend must be one of"):
                server.normalize_backend_settings(
                    "embedding.gguf",
                    self.model,
                    {"backend": "cuda", "mode": "auto", "pooling": "auto"},
                )

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

    async def test_default_backend_is_used_before_latest_for_unspecified_model(self) -> None:
        registry = server.BackendRegistry()

        class FakeInstance:
            def __init__(self, model_id: str, started_at: float) -> None:
                self.model_id = model_id
                self.effective_mode = "chat"
                self.started_at = started_at
                self.backend_url = f"http://{model_id}"
                self.last_used_at = None

            def is_active(self) -> bool:
                return True

        default = FakeInstance("default.gguf", 10)
        latest = FakeInstance("latest.gguf", 20)
        registry.instances = {"default.gguf": default, "latest.gguf": latest}
        registry.default_model_ids = {"chat": "default.gguf"}

        with patch("llm_server.backend.wait_for_backend", AsyncMock(return_value=True)):
            selected, error = await registry.backend_for_request(None, purpose="chat")

        self.assertIsNone(error)
        self.assertIs(selected, default)

    async def test_set_default_model_infers_running_model_purpose(self) -> None:
        registry = server.BackendRegistry()
        registry.default_model_ids = {}

        class FakeInstance:
            model_id = "embedding.gguf"
            effective_mode = "embeddings"

            def is_active(self) -> bool:
                return True

        registry.instances = {"embedding.gguf": FakeInstance()}

        with (
            patch(
                "llm_server.backend.resolve_model_reference_required",
                return_value=("embedding.gguf", Path("/tmp/embedding.gguf")),
            ),
            patch.object(registry, "_persist_settings"),
            patch.object(registry, "status", AsyncMock(return_value={})),
        ):
            result = await registry.set_default_model({"model": "embedding.gguf"})

        self.assertTrue(result["ok"])
        self.assertEqual(registry.default_model_ids, {"embeddings": "embedding.gguf"})

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

    def test_embedding_exchange_is_logged_by_model(self) -> None:
        instance = type("Instance", (), {"model_id": "nested/embedding.gguf", "backend_url": "http://backend"})()
        backend_response = httpx.Response(
            200,
            json={
                "object": "list",
                "model": "nested/embedding.gguf",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            },
        )

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json, headers):
                return backend_response

        with tempfile.TemporaryDirectory() as log_dir:
            request_logger = RequestResponseLogger(Path(log_dir), retention_days=7)
            with (
                patch.object(
                    server.registry,
                    "backend_for_request",
                    AsyncMock(return_value=(instance, None)),
                ),
                patch.object(server.httpx, "AsyncClient", return_value=FakeClient()),
                patch("llm_server.openai_api.request_logger", request_logger),
                TestClient(server.app) as client,
            ):
                response = client.post(
                    "/v1/embeddings",
                    json={"model": "local", "input": ["hello"], "encoding_format": "float"},
                )

            self.assertEqual(response.status_code, 200)
            log_files = list(Path(log_dir).glob("*.jsonl"))
            self.assertEqual(len(log_files), 1)
            self.assertIn("nested_embedding.gguf", log_files[0].name)
            record = json.loads(log_files[0].read_text(encoding="utf-8").strip())
            self.assertEqual(record["endpoint"], "/v1/embeddings")
            self.assertEqual(record["model"], "nested/embedding.gguf")
            self.assertEqual(record["request"]["model"], "nested/embedding.gguf")
            self.assertEqual(record["request"]["input"], ["hello"])
            self.assertEqual(record["response"]["status_code"], 200)
            self.assertEqual(record["response"]["body"]["data"][0]["embedding"], [0.1, 0.2])


class RequestResponseLoggerTests(unittest.TestCase):
    def test_old_daily_logs_are_deleted_on_write(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            root = Path(log_dir)
            today = date.today()
            old_log = root / f"model-000000000000.{(today - timedelta(days=8)).isoformat()}.jsonl"
            kept_log = root / f"model-000000000000.{(today - timedelta(days=6)).isoformat()}.jsonl"
            old_log.write_text("{}\n", encoding="utf-8")
            kept_log.write_text("{}\n", encoding="utf-8")

            request_logger = RequestResponseLogger(root, retention_days=7)
            request_logger.log(
                request_id="req",
                endpoint="/v1/chat/completions",
                model_id="model.gguf",
                request_payload={"model": "model.gguf"},
                status_code=200,
                response_body={"ok": True},
                started_at=0,
                completed_at=0,
            )

            self.assertFalse(old_log.exists())
            self.assertTrue(kept_log.exists())


class WebUiTests(unittest.TestCase):
    def test_ui_and_static_assets_are_served(self) -> None:
        with TestClient(server.app) as client:
            index = client.get("/")
            stylesheet = client.get("/static/app.css")
            script = client.get("/static/app.js")

        self.assertEqual(index.status_code, 200)
        self.assertIn('href="/static/app.css"', index.text)
        self.assertIn('src="/static/app.js"', index.text)
        self.assertIn('id="settingsDialog"', index.text)
        self.assertIn('id="modelRows"', index.text)
        self.assertIn('id="defaultChatModel"', index.text)
        self.assertIn('id="mtp"', index.text)
        self.assertIn('id="mtp_draft_tokens"', index.text)
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn(".settings-dialog", stylesheet.text)
        self.assertEqual(script.status_code, 200)
        self.assertIn("function connectLogStream()", script.text)
        self.assertIn("function updateModelRow(", script.text)
        self.assertIn("function setDefaultModel(", script.text)
        self.assertIn("/api/default-model", script.text)
        self.assertIn("function updateMtpControl()", script.text)
        self.assertNotIn("modelRows.innerHTML", script.text)
        self.assertNotIn("recentRows.innerHTML", script.text)
        self.assertNotIn("backendRows.innerHTML", script.text)
        self.assertIn("setInterval(loadStatus, 5000)", script.text)


if __name__ == "__main__":
    unittest.main()

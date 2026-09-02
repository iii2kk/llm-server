from __future__ import annotations

import json
import os
import struct
import tempfile
import time
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

import server
from llm_server.request_logs import (
    RequestResponseLogger,
    decode_response_body,
    read_request_logs,
    request_log_options,
)


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

    def test_llama_bin_dir_accepts_build_root_or_bin_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "build-cuda"
            bin_dir = build_dir / "bin"
            bin_dir.mkdir(parents=True)
            llama_server = bin_dir / "llama-server"
            llama_server.write_text("#!/bin/sh\n", encoding="ascii")
            llama_server.chmod(0o755)

            self.assertEqual(server.find_llama_bin_dir(str(build_dir)), bin_dir)
            self.assertEqual(server.find_llama_bin_dir(str(bin_dir)), bin_dir)

    def test_unavailable_llama_backends_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "build-cuda"
            bin_dir = build_dir / "bin"
            bin_dir.mkdir(parents=True)
            llama_server = bin_dir / "llama-server"
            llama_server.write_text("#!/bin/sh\n", encoding="ascii")
            llama_server.chmod(0o755)

            available = server.available_llama_bin_dirs(
                {
                    "cuda": str(build_dir),
                    "rocm": str(Path(temp_dir) / "missing-rocm"),
                    "rocm-fastmtp": "",
                }
            )

            self.assertEqual(available, {"cuda": bin_dir})


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
        with patch.object(server, "LLAMA_BIN_DIRS", {"cuda": Path("/cuda")}):
            settings = server.normalize_backend_settings(
                "embedding.gguf",
                self.model,
                {"backend": "cuda", "mode": "auto", "pooling": "auto"},
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

    def test_cuda_command_does_not_enable_direct_io(self) -> None:
        fake_bin_dir = self.root / "cuda-bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        command = server.build_llama_command(
            {"backend": "cuda"},
            model=self.model,
            port=9999,
            llama_bin_dir=fake_bin_dir,
        )

        self.assertNotIn("--direct-io", command)

    def test_fastmtp_rocm_command_enables_direct_io(self) -> None:
        fake_bin_dir = self.root / "fastmtp-rocm-bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        command = server.build_llama_command(
            {"backend": "rocm-fastmtp"},
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

    def test_reasoning_effort_is_validated_and_added_to_command(self) -> None:
        fake_bin_dir = self.root / "reasoning-bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        command = server.build_llama_command(
            {"reasoning_effort": "low"},
            model=self.model,
            port=9999,
            llama_bin_dir=fake_bin_dir,
        )
        self.assertEqual(command[command.index("--reasoning-effort") + 1], "low")

        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            server.build_llama_command(
                {"reasoning_effort": "extreme"},
                model=self.model,
                port=9999,
                llama_bin_dir=fake_bin_dir,
            )

    def test_reasoning_preserve_is_added_to_command(self) -> None:
        fake_bin_dir = self.root / "reasoning-preserve-bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        enabled = server.build_llama_command(
            {"reasoning_preserve": True},
            model=self.model,
            port=9999,
            llama_bin_dir=fake_bin_dir,
        )
        disabled = server.build_llama_command(
            {"reasoning_preserve": False},
            model=self.model,
            port=9999,
            llama_bin_dir=fake_bin_dir,
        )

        self.assertIn("--reasoning-preserve", enabled)
        self.assertIn("--no-reasoning-preserve", disabled)

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

    def test_external_mtp_draft_model_is_detected_and_added_to_command(self) -> None:
        model = self.root / "gemma4.gguf"
        draft_model = self.root / "mtp-gemma4.gguf"
        write_gguf(model, [("general.architecture", 8, "gemma4")])
        write_gguf(
            draft_model,
            [
                ("general.architecture", 8, "gemma4-assistant"),
                ("gemma4-assistant.nextn_predict_layers", 4, 4),
            ],
        )

        settings = server.normalize_backend_settings(
            "gemma4.gguf",
            model,
            {"mtp": "auto", "mtp_draft_tokens": 3, "gpu_layers": "all"},
        )

        self.assertTrue(settings["effective_mtp"])
        self.assertEqual(settings["mtp_type"], "external")
        self.assertEqual(settings["mtp_draft_path"], str(draft_model.resolve()))

        fake_bin_dir = self.root / "external-mtp-bin"
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
        self.assertEqual(command[command.index("--spec-draft-model") + 1], str(draft_model.resolve()))
        self.assertEqual(command[command.index("--spec-draft-ngl") + 1], "all")

    def test_fastmtp_sidecar_requires_fastmtp_backend(self) -> None:
        model = self.root / "qwen.gguf"
        draft_model = self.root / "Qwen-FastMTP-32K.gguf"
        fields = [
            ("general.architecture", 8, "qwen35"),
            ("qwen35.nextn_predict_layers", 4, 1),
        ]
        write_gguf(model, fields)
        write_gguf(draft_model, fields)
        fake_bin_dir = self.root / "fastmtp-bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        with patch.object(
            server,
            "LLAMA_BIN_DIRS",
            {"rocm": fake_bin_dir, "rocm-fastmtp": fake_bin_dir},
        ):
            upstream = server.normalize_backend_settings(
                "qwen.gguf",
                model,
                {"backend": "rocm", "mtp": "auto", "gpu_layers": "all"},
            )
            fastmtp = server.normalize_backend_settings(
                "qwen.gguf",
                model,
                {"backend": "rocm-fastmtp", "mtp": "auto", "gpu_layers": "all"},
            )

        self.assertEqual(upstream["mtp_type"], "embedded")
        self.assertEqual(upstream["mtp_draft_path"], "")
        self.assertEqual(fastmtp["mtp_type"], "fastmtp")
        self.assertEqual(fastmtp["mtp_draft_path"], str(draft_model.resolve()))
        self.assertTrue(server.is_mtp_draft_file(draft_model))

        command = server.build_llama_command(
            fastmtp,
            model=model,
            port=9999,
            llama_bin_dir=fake_bin_dir,
        )
        self.assertEqual(command[command.index("--spec-draft-model") + 1], str(draft_model.resolve()))
        self.assertEqual(command[command.index("--spec-draft-ngl") + 1], "all")
        self.assertEqual(command[command.index("--spec-draft-p-min") + 1], "0")

    def test_qwen4exp_sidecar_requires_rocmfpx_backend(self) -> None:
        model = self.root / "Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf"
        draft_model = self.root / "Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST.gguf"
        write_gguf(model, [("general.architecture", 8, "qwen4exp")])
        write_gguf(
            draft_model,
            [
                ("general.architecture", 8, "qwen4exp"),
                ("qwen4exp.nextn_predict_layers", 4, 1),
            ],
        )
        fake_bin_dir = self.root / "rocmfpx-bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        with patch.object(
            server,
            "LLAMA_BIN_DIRS",
            {"vulkan": fake_bin_dir, "vulkan-rocmfpx": fake_bin_dir},
        ):
            upstream = server.normalize_backend_settings(
                model.name,
                model,
                {"backend": "vulkan", "mtp": "auto"},
            )
            rocmfpx = server.normalize_backend_settings(
                model.name,
                model,
                {
                    "backend": "vulkan-rocmfpx",
                    "mtp": "auto",
                    "gpu_layers": "all",
                    "cache_type_k": "q8_0",
                    "cache_type_v": "q8_0",
                    "flash_attn": "on",
                },
            )

        self.assertFalse(upstream["effective_mtp"])
        self.assertEqual(rocmfpx["mtp_type"], "qwen4exp-external")
        self.assertEqual(rocmfpx["mtp_draft_path"], str(draft_model.resolve()))
        self.assertEqual(rocmfpx["mtp_draft_tokens"], 4)
        self.assertTrue(server.is_mtp_draft_file(draft_model))

        command = server.build_llama_command(
            rocmfpx,
            model=model,
            port=9999,
            llama_bin_dir=fake_bin_dir,
        )
        self.assertNotIn("--direct-io", command)
        self.assertEqual(command[command.index("--spec-draft-model") + 1], str(draft_model.resolve()))
        self.assertEqual(command[command.index("--spec-draft-ngl") + 1], "all")
        self.assertEqual(command[command.index("--spec-draft-n-min") + 1], "2")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "4")
        self.assertIn("--spec-draft-adaptive", command)
        self.assertEqual(command[command.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(command[command.index("--cache-type-v") + 1], "q8_0")

        with (
            patch("llm_server.models.MODEL_DIR", self.root),
            patch("llm_server.models.LLAMA_BIN_DIRS", {"vulkan-rocmfpx": fake_bin_dir}),
            patch("llm_server.models.DEFAULT_LLAMA_BACKEND", "vulkan-rocmfpx"),
        ):
            options = server.model_options(
                {model.name: {"backend": "vulkan-rocmfpx", "mtp": "auto"}}
            )
            resolved_draft = server.resolve_model_reference(draft_model.name, {})

        names = {item["name"] for item in options}
        self.assertIn(model.name, names)
        self.assertNotIn(draft_model.name, names)
        self.assertIsNone(resolved_draft)

    def test_cache_types_are_validated(self) -> None:
        fake_bin_dir = self.root / "cache-bin"
        fake_bin_dir.mkdir()
        llama_server = fake_bin_dir / "llama-server"
        llama_server.write_text("#!/bin/sh\n", encoding="ascii")
        llama_server.chmod(0o755)

        with self.assertRaisesRegex(ValueError, "cache_type_k"):
            server.build_llama_command(
                {"cache_type_k": "not-a-cache-type"},
                model=self.model,
                port=9999,
                llama_bin_dir=fake_bin_dir,
            )

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
            {"cuda": Path("/cuda"), "vulkan": Path("/vulkan"), "rocm": Path("/rocm")},
        ):
            settings = server.normalize_backend_settings(
                "embedding.gguf",
                self.model,
                {"backend": "rocm", "mode": "auto", "pooling": "auto"},
            )
            self.assertEqual(settings["backend"], "rocm")

            settings = server.normalize_backend_settings(
                "embedding.gguf",
                self.model,
                {"backend": "cuda", "mode": "auto", "pooling": "auto"},
            )
            self.assertEqual(settings["backend"], "cuda")

            with self.assertRaisesRegex(ValueError, "backend must be one of"):
                server.normalize_backend_settings(
                    "embedding.gguf",
                    self.model,
                    {"backend": "metal", "mode": "auto", "pooling": "auto"},
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

    async def test_startup_profile_save_uses_active_instances_and_defaults(self) -> None:
        registry = server.BackendRegistry()
        registry.default_model_ids = {"chat": "chat.gguf"}

        class FakeInstance:
            def __init__(self, model_id: str, started_at: float, active: bool = True) -> None:
                self.model_id = model_id
                self.started_at = started_at
                self.settings = {"model": model_id, "backend": "rocm", "ctx_size": 4096}
                self._active = active

            def is_active(self) -> bool:
                return self._active

        registry.instances = {
            "stopped.gguf": FakeInstance("stopped.gguf", 0, active=False),
            "second.gguf": FakeInstance("second.gguf", 20),
            "chat.gguf": FakeInstance("chat.gguf", 10),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = await registry.save_startup_profile(Path(temp_dir) / "startup.json")
            payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertEqual([item["model"] for item in payload["models"]], ["chat.gguf", "second.gguf"])
        self.assertEqual(payload["default_models"], {"chat": "chat.gguf"})
        self.assertEqual(payload["models"][0]["ctx_size"], 4096)

    async def test_startup_profile_load_starts_models_and_applies_defaults(self) -> None:
        registry = server.BackendRegistry()

        with tempfile.TemporaryDirectory() as temp_dir:
            profile = Path(temp_dir) / "startup.json"
            profile.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "default_models": {"chat": "chat.gguf", "embeddings": "embed.gguf"},
                        "models": [
                            {"model": "chat.gguf", "backend": "vulkan"},
                            {"model": "embed.gguf", "mode": "embeddings"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            start_mock = AsyncMock(return_value={"ok": True, "backend_reachable": True})
            with (
                patch.object(registry, "start", start_mock),
                patch.object(registry, "_persist_settings"),
            ):
                result = await registry.load_startup_profile(profile)

        self.assertTrue(result["ok"])
        self.assertEqual(result["loaded"], 2)
        self.assertEqual(registry.default_model_ids, {"chat": "chat.gguf", "embeddings": "embed.gguf"})
        start_mock.assert_any_await({"model": "chat.gguf", "backend": "vulkan"}, conflict_if_running=False)
        start_mock.assert_any_await({"model": "embed.gguf", "mode": "embeddings"}, conflict_if_running=False)

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
    def test_stream_response_body_is_decoded_to_final_message(self) -> None:
        body = (
            b'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"thinking"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
            b"data: [DONE]\n\n"
        )

        decoded = decode_response_body(body, "text/event-stream")

        self.assertEqual(decoded["stream"], True)
        self.assertEqual(decoded["done"], True)
        self.assertEqual(decoded["chunks"], 3)
        self.assertEqual(decoded["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(decoded["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(decoded["choices"][0]["message"]["reasoning_content"], "thinking")
        self.assertEqual(decoded["choices"][0]["finish_reason"], "stop")
        self.assertEqual(decoded["usage"]["total_tokens"], 5)

    def test_existing_stream_logs_are_decoded_before_compaction(self) -> None:
        sse = (
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"hello "}}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )

        with tempfile.TemporaryDirectory() as log_dir:
            root = Path(log_dir)
            request_logger = RequestResponseLogger(root, retention_days=7)
            now = time.time()
            request_logger.log(
                request_id="stream",
                endpoint="/v1/chat/completions",
                model_id="model.gguf",
                request_payload={"model": "model.gguf", "stream": True},
                status_code=200,
                response_body=sse,
                started_at=now,
                completed_at=now,
                stream=True,
            )

            data = read_request_logs(log_dir=root)
            body = data["records"][0]["record"]["response"]["body"]
            self.assertEqual(body["choices"][0]["message"]["content"], "hello world")
            self.assertNotIn("omitted", body)

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

    def test_request_log_options_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            root = Path(log_dir)
            request_logger = RequestResponseLogger(root, retention_days=7)
            now = time.time()
            request_logger.log(
                request_id="ok",
                endpoint="/v1/chat/completions",
                model_id="model.gguf",
                request_payload={"model": "model.gguf", "messages": [{"role": "user", "content": "hello"}]},
                status_code=200,
                response_body={"choices": [{"message": {"content": "hi"}}]},
                started_at=now - 0.1,
                completed_at=now,
            )
            request_logger.log(
                request_id="error",
                endpoint="/v1/embeddings",
                model_id="embedding.gguf",
                request_payload={"model": "embedding.gguf", "input": ["hello"]},
                status_code=500,
                response_body={"error": {"message": "failed"}},
                started_at=now - 0.2,
                completed_at=now + 1,
                error="backend_error_response",
            )

            options = request_log_options(root)
            self.assertEqual(options["total"], 2)
            self.assertIn("/v1/chat/completions", options["endpoints"])
            self.assertEqual(options["status_counts"], {"success": 1, "error": 1})

            data = read_request_logs(log_dir=root, status="success", query="hello")
            self.assertEqual(data["total"], 1)
            record = data["records"][0]["record"]
            self.assertEqual(record["model"], "model.gguf")
            self.assertEqual(record["status"], "success")

    def test_request_log_reader_compacts_large_numeric_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            root = Path(log_dir)
            request_logger = RequestResponseLogger(root, retention_days=7)
            now = time.time()
            request_logger.log(
                request_id="embedding",
                endpoint="/v1/embeddings",
                model_id="embedding.gguf",
                request_payload={"model": "embedding.gguf", "input": ["hello"]},
                status_code=200,
                response_body={
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": [float(index) for index in range(128)],
                        }
                    ],
                },
                started_at=now,
                completed_at=now,
            )

            data = read_request_logs(log_dir=root)
            embedding = data["records"][0]["record"]["response"]["body"]["data"][0]["embedding"]
            self.assertEqual(embedding["omitted"], "numeric_array")
            self.assertEqual(embedding["length"], 128)


class WebUiTests(unittest.TestCase):
    def test_ui_and_static_assets_are_served(self) -> None:
        with TestClient(server.app) as client:
            index = client.get("/")
            request_logs = client.get("/request-logs")
            stylesheet = client.get("/static/app.css")
            script = client.get("/static/app.js")
            request_logs_script = client.get("/static/request-logs.js")

        self.assertEqual(index.status_code, 200)
        self.assertEqual(request_logs.status_code, 200)
        self.assertIn('href="/static/app.css?v=', index.text)
        self.assertIn('src="/static/app.js?v=', index.text)
        self.assertIn('href="/request-logs"', index.text)
        self.assertIn('id="requestLogRows"', request_logs.text)
        self.assertIn('src="/static/request-logs.js?v=', request_logs.text)
        self.assertIn('id="settingsDialog"', index.text)
        self.assertIn('id="modelRows"', index.text)
        self.assertNotIn('id="requestLogRows"', index.text)
        self.assertIn('id="defaultChatModel"', index.text)
        self.assertIn('id="startupProfilePath"', index.text)
        self.assertIn('id="mtp"', index.text)
        self.assertIn('id="mtp_draft_tokens"', index.text)
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn(".settings-dialog", stylesheet.text)
        self.assertIn(".request-log-table", stylesheet.text)
        self.assertEqual(script.status_code, 200)
        self.assertEqual(request_logs_script.status_code, 200)
        self.assertIn("function connectLogStream()", script.text)
        self.assertIn("function loadRequestLogs(", request_logs_script.text)
        self.assertIn("/api/request-logs", request_logs_script.text)
        self.assertIn("function updateModelRow(", script.text)
        self.assertIn("function setDefaultModel(", script.text)
        self.assertIn("/api/default-model", script.text)
        self.assertIn("function saveStartupProfile(", script.text)
        self.assertIn("/api/startup-profile", script.text)
        self.assertIn("function updateMtpControl()", script.text)
        self.assertNotIn("modelRows.innerHTML", script.text)
        self.assertNotIn("recentRows.innerHTML", script.text)
        self.assertNotIn("backendRows.innerHTML", script.text)
        self.assertIn("setInterval(loadStatus, 5000)", script.text)


if __name__ == "__main__":
    unittest.main()

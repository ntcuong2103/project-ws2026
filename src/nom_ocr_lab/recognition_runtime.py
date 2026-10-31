"""Configured recognizer access for Weeks 6--7.

The public runtime deliberately exposes logits, not a model-provided search
method.  Students use :meth:`next_token_logits` to build greedy decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Protocol

import yaml
from PIL import Image, ImageOps

from .recognition import load_expanded_ids, load_vocabulary


@dataclass(frozen=True)
class RecognitionConfig:
    paths: dict[str, Path]
    model: dict[str, Any]
    preprocessing: dict[str, Any]
    inference: dict[str, Any]

    @property
    def sos_idx(self) -> int:
        return int(self.model["sos_idx"])

    @property
    def eos_idx(self) -> int:
        return int(self.model["eos_idx"])


@dataclass(frozen=True)
class RecognitionCrop:
    """Crop metadata produced by a team's implementation."""

    page_id: str
    box_id: int
    image_path: Path
    class_id: int
    confidence: float | None
    normalized_box: tuple[float, float, float, float]
    pixel_box: tuple[int, int, int, int]
    expanded_box: tuple[int, int, int, int]
    image: Image.Image


class LogitBackend(Protocol):
    def logits(self, crops: Any, prefixes: Any) -> Any:
        """Return next-token or teacher-forced logits with vocabulary last."""


def _expanded(value: object) -> object:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expanded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expanded(item) for item in value]
    return value


def load_config(path: Path, *, require_files: bool = True) -> RecognitionConfig:
    """Load the YAML contract and reject incomplete or incompatible assets."""

    payload = _expanded(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    required_paths = {
        "detector_checkpoint", "recognizer_checkpoint", "vocabulary", "expanded_ids",
        "images", "detector_labels", "outputs",
    }
    for section in ("paths", "model", "preprocessing", "inference"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"{path}: missing mapping {section!r}")
    missing = required_paths - set(payload["paths"])
    if missing:
        raise ValueError(f"{path}: missing paths: {', '.join(sorted(missing))}")
    for key in ("sos_idx", "eos_idx", "pad_idx", "class_path"):
        if key not in payload["model"]:
            raise ValueError(f"{path}: model.{key} is required")
    if int(payload["model"]["sos_idx"]) == int(payload["model"]["eos_idx"]):
        raise ValueError("SOS and EOS token IDs must differ")
    paths = {key: Path(str(value)).expanduser() for key, value in payload["paths"].items()}
    config = RecognitionConfig(paths, dict(payload["model"]), dict(payload["preprocessing"]), dict(payload["inference"]))
    if require_files:
        for key in ("detector_checkpoint", "recognizer_checkpoint", "vocabulary", "expanded_ids", "images", "detector_labels"):
            if not config.paths[key].exists():
                raise FileNotFoundError(f"Configured {key} does not exist: {config.paths[key]}")
        vocabulary = load_vocabulary(config.paths["vocabulary"])
        load_expanded_ids(config.paths["expanded_ids"])
        expected = config.model.get("vocab_size")
        checkpoint_vocabulary_size = len(vocabulary) + 3  # PAD, SOS, EOS
        if expected is not None and int(expected) != checkpoint_vocabulary_size:
            raise ValueError(
                f"Configured vocab_size {expected} does not match {checkpoint_vocabulary_size} "
                "tokens (base vocabulary plus PAD/SOS/EOS)"
            )
    return config


def preprocess(crop: RecognitionCrop, config: RecognitionConfig) -> Any:
    """Resize and invert deterministically; no random crop is used in inference."""

    import numpy as np
    import torch

    size = int(config.preprocessing.get("image_size", 128))
    image = crop.image.resize((size, size), Image.Resampling.BILINEAR)
    if bool(config.preprocessing.get("invert", True)):
        image = ImageOps.invert(image)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255)


class RecognizerRuntime:
    """Thin adapter around the supplied LitBTTR model or a test backend."""

    def __init__(self, config: RecognitionConfig, backend: LogitBackend | None = None):
        self.config = config
        self.backend = backend or self._load_litbttr()

    def _load_litbttr(self) -> LogitBackend:
        package_root = self.config.model.get("package_root")
        if package_root:
            package_path = str(Path(str(package_root)).expanduser())
            if package_path not in sys.path:
                sys.path.insert(0, package_path)
        module_name, separator, class_name = str(self.config.model["class_path"]).partition(":")
        if not separator:
            raise ValueError("model.class_path must be module:Class")
        model_class = getattr(importlib.import_module(module_name), class_name)
        kwargs = {
            key: value for key, value in self.config.model.items()
            if key in {"d_model", "growth_rate", "num_layers", "nhead", "num_decoder_layers", "dim_feedforward", "dropout", "vocab_size"} and value is not None
        }
        kwargs.update({"SOS_IDX": self.config.sos_idx, "EOS_IDX": self.config.eos_idx, "PAD_IDX": int(self.config.model["pad_idx"])})
        model = model_class.load_from_checkpoint(str(self.config.paths["recognizer_checkpoint"]), map_location=self.config.inference.get("device", "cpu"), **kwargs)
        expected = self.config.model.get("vocab_size")
        projection = getattr(getattr(getattr(model, "bttr", None), "decoder", None), "proj", None)
        actual = getattr(projection, "out_features", None)
        if expected is not None and actual is not None and int(expected) != int(actual):
            raise ValueError(
                f"Checkpoint output vocabulary {actual} does not match configured vocab_size {expected}"
            )
        model.eval()
        return _LitBTTRBackend(model)

    def teacher_forced_logits(self, crops: Any, prefixes: Any) -> Any:
        """Return logits shaped ``[batch, step, vocabulary]`` for supplied prefixes."""
        logits = self.backend.logits(crops, prefixes)
        if getattr(logits, "ndim", None) != 3:
            raise ValueError("Recognizer backend must return [batch, step, vocabulary] logits")
        return logits

    def next_token_logits(self, crops: Any, prefixes: Any) -> Any:
        """Return ``[batch, vocabulary]`` logits for each current decoder prefix."""
        return self.teacher_forced_logits(crops, prefixes)[:, -1, :]


class _LitBTTRBackend:
    def __init__(self, model: Any):
        self.model = model

    def logits(self, crops: Any, prefixes: Any) -> Any:
        """Adapt the pinned LitBTTR teacher-forced forward method to l2r logits."""
        import torch

        if crops.ndim != 4 or prefixes.ndim != 2 or crops.shape[0] != prefixes.shape[0]:
            raise ValueError("Expected crops [batch, channels, height, width] and prefixes [batch, step]")
        device = next(self.model.parameters()).device
        crops = crops.to(device)
        prefixes = prefixes.to(device)
        image_mask = torch.zeros(
            (crops.shape[0], crops.shape[2], crops.shape[3]), dtype=torch.long, device=device
        )
        # LitBTTR is bidirectionally trained and expects 2B target rows. The
        # first B output rows are the ordinary left-to-right teacher-forced logits.
        bidirectional_prefixes = torch.cat((prefixes, prefixes), dim=0)
        with torch.no_grad():
            return self.model(crops, image_mask, bidirectional_prefixes)[: crops.shape[0]]


def _token_name(token_id: int, vocabulary: list[str]) -> str | None:
    special = {0: "<pad>", 1: "<sos>", 2: "<eos>"}
    if token_id in special:
        return special[token_id]
    index = token_id - 3
    return vocabulary[index] if 0 <= index < len(vocabulary) else None


def write_top_token_report(path: Path, crops: Iterable[RecognitionCrop], logits: Any, vocabulary: list[str], k: int = 5) -> None:
    """Write a compact, inspectable Week 6 logits report."""
    rows = []
    for crop, row in zip(crops, logits):
        top = row[-1].topk(k)
        rows.append({"page_id": crop.page_id, "box_id": crop.box_id, "top_tokens": [{"token_id": int(index), "token": _token_name(int(index), vocabulary), "logit": float(value)} for value, index in zip(top.values, top.indices)]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

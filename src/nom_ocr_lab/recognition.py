"""IDS resources and text-level conversion helpers for the recognition labs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import BinaryIO, Callable
from urllib.request import urlopen


NOM_IDS_COMMIT = "6f7f7f8242e95a95eb6920092cd290d9e4e49deb"
CJKVI_COMMIT = "86b4d16159f0079437870408f0ca186e529015db"


@dataclass(frozen=True)
class ResourceSpec:
    repository: str
    commit: str
    relative_path: str
    url: str
    sha256: str


RESOURCE_SPECS = (
    ResourceSpec(
        repository="ntcuong2103/nom-ids",
        commit=NOM_IDS_COMMIT,
        relative_path="nom-ids/ids_exp.txt",
        url=(
            "https://raw.githubusercontent.com/ntcuong2103/nom-ids/"
            f"{NOM_IDS_COMMIT}/ids_exp.txt"
        ),
        sha256="9097ebdc17fc684c77b7a4c411afe28a7035671aa3eae7c91d0881753e2c37a0",
    ),
    ResourceSpec(
        repository="ntcuong2103/nom-ids",
        commit=NOM_IDS_COMMIT,
        relative_path="nom-ids/vocab_ids.txt",
        url=(
            "https://raw.githubusercontent.com/ntcuong2103/nom-ids/"
            f"{NOM_IDS_COMMIT}/vocab_ids.txt"
        ),
        sha256="2e71480db4467ce907375f18219c3d2672c6cb1a8066760206fb17dcbc328453",
    ),
    ResourceSpec(
        repository="cjkvi/cjkvi-ids",
        commit=CJKVI_COMMIT,
        relative_path="cjkvi/ids.txt",
        url=(
            "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/"
            f"{CJKVI_COMMIT}/ids.txt"
        ),
        sha256="bfc70a8c09f9f5616ebf0543bd6681e67314e9f7ae2307e5ae8c6f15bdc5c6a6",
    ),
    ResourceSpec(
        repository="cjkvi/cjkvi-ids",
        commit=CJKVI_COMMIT,
        relative_path="cjkvi/ids-ext-cdef.txt",
        url=(
            "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/"
            f"{CJKVI_COMMIT}/ids-ext-cdef.txt"
        ),
        sha256="5712a8e4453db5d908399436c344e753a686365d4ffecc9011d43447d95db0e7",
    ),
)


@dataclass(frozen=True)
class CharacterIDS:
    character: str
    codepoint: str
    ids: str | None
    tokens: tuple[str, ...]
    status: str

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["tokens"] = list(self.tokens)
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_resources(
    output: Path,
    *,
    opener: Callable[[str], BinaryIO] = urlopen,
) -> dict[str, object]:
    """Download pinned IDS files, verify them, and write a provenance manifest."""

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for spec in RESOURCE_SPECS:
        destination = output / spec.relative_path
        reused = destination.exists()
        if reused:
            actual = sha256_file(destination)
            if actual != spec.sha256:
                raise FileExistsError(
                    f"Refusing to replace checksum-mismatched resource: {destination}"
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".part")
            if partial.exists():
                raise FileExistsError(f"Refusing to replace partial download: {partial}")
            try:
                with opener(spec.url) as response, partial.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                actual = sha256_file(partial)
                if actual != spec.sha256:
                    raise ValueError(
                        f"Checksum mismatch for {spec.url}: expected {spec.sha256}, got {actual}"
                    )
                partial.replace(destination)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        entries.append(
            {
                "path": spec.relative_path,
                "repository": spec.repository,
                "commit": spec.commit,
                "url": spec.url,
                "sha256": spec.sha256,
                "bytes": destination.stat().st_size,
                "reused": reused,
            }
        )

    manifest = {"version": 1, "kind": "ids-resources", "resources": entries}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Re-running is allowed when the downloaded resource identity is unchanged.
        def identity(payload: dict[str, object]) -> list[tuple[object, ...]]:
            return [
                (item.get("path"), item.get("commit"), item.get("sha256"), item.get("bytes"))
                for item in payload.get("resources", [])
            ]

        if identity(existing) != identity(manifest):
            raise FileExistsError(f"Refusing to replace conflicting manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_expanded_ids(path: Path) -> dict[str, str]:
    """Load the authoritative ``character<TAB>expanded IDS`` mapping."""

    mapping: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            continue
        fields = raw.split("\t")
        if len(fields) != 2 or len(fields[0]) != 1 or not fields[1]:
            raise ValueError(f"{path}:{line_number}: expected character<TAB>expanded IDS")
        if fields[0] in mapping:
            raise ValueError(f"{path}:{line_number}: duplicate character {fields[0]!r}")
        mapping[fields[0]] = fields[1]
    return mapping


def load_vocabulary(path: Path) -> list[str]:
    """Load token strings without sorting or otherwise changing checkpoint order."""

    tokens = path.read_text(encoding="utf-8").splitlines()
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{path}: vocabulary must contain non-empty tokens")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"{path}: vocabulary contains duplicate tokens")
    return tokens


def load_cjkvi_ids(path: Path) -> dict[str, str]:
    """Load the first direct IDS per CJKVI row and remove region annotations."""

    mapping: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) < 3 or len(fields[1]) != 1:
            raise ValueError(f"{path}:{line_number}: malformed CJKVI IDS row")
        mapping[fields[1]] = re.sub(r"\[[^]]*\]", "", fields[2])
    return mapping


def reverse_ids(mapping: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """Return every Unicode character associated with each IDS string."""

    reverse: defaultdict[str, list[str]] = defaultdict(list)
    for character, ids in mapping.items():
        reverse[ids].append(character)
    return {ids: tuple(characters) for ids, characters in reverse.items()}


def text_to_ids(text: str, mapping: dict[str, str]) -> list[CharacterIDS]:
    """Convert non-whitespace Unicode code points to structured IDS records."""

    records = []
    for character in text:
        if character.isspace():
            continue
        ids = mapping.get(character)
        records.append(
            CharacterIDS(
                character=character,
                codepoint=f"U+{ord(character):04X}",
                ids=ids,
                tokens=tuple(ids) if ids is not None else (),
                status="covered" if ids is not None else "oov",
            )
        )
    return records

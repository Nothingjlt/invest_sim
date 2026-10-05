"""Explicit labels for the source of simulated returns, independent of sampler.

Historical provenance is a caller declaration, not a claim that this package
has independently audited a dataset. CSV digests bind that declaration to the
input bytes. Unidentified inputs stay unknown, regardless of their filename.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
import re
from typing import Literal


@dataclass(frozen=True)
class DataProvenance:
    """Immutable source metadata carried by markets and simulation results."""

    kind: Literal["synthetic", "historical", "unknown"] = "unknown"
    source: str | None = None
    sha256: str | None = None

    def __post_init__(self):
        if self.kind not in ("synthetic", "historical", "unknown"):
            raise ValueError("Provenance kind must be synthetic, historical, or unknown.")
        if self.source is not None and (
            not isinstance(self.source, str) or not self.source.strip()
        ):
            raise ValueError("Provenance source must be a nonempty description.")
        if self.kind != "unknown" and self.source is None:
            raise ValueError("Synthetic and historical provenance require a source.")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise ValueError("Provenance sha256 must be a lowercase SHA-256 hex digest.")

    @property
    def label(self) -> str:
        """Readable label suitable for table headings and plot titles."""
        labels = {
            "synthetic": "Synthetic/generated data",
            "historical": "Historical data (declared source)",
            "unknown": "Unknown/unverified data",
        }
        label = labels[self.kind]
        return f"{label}: {self.source}" if self.source else label

    def to_dict(self) -> dict:
        """JSON-compatible metadata for saved results."""
        return asdict(self)


# Explicit metadata for the checked-in generated panel. Recognition is by
# content, so copies retain their label and similarly named files cannot inherit
# it. Regenerated/custom panels should supply their own provenance declaration.
_BUNDLED_GENERATED_SHA256 = (
    "28842fd15a2c33264e8c30c95d6d245dee43445189406ae4b087ed910dd025d9"
)
_BUNDLED_GENERATED_SOURCE = "generate_global_data.py (bundled generated panel)"


def csv_provenance(
    csv_path: str | Path, declared: DataProvenance | None = None
) -> DataProvenance:
    """Bind explicit source metadata to a CSV, or mark its origin unverified.

    Opening the input here intentionally fails before any loader can substitute
    generated observations for a missing file.
    """
    if declared is not None and not isinstance(declared, DataProvenance):
        raise TypeError("provenance must be a DataProvenance instance.")
    digest = hashlib.sha256()
    with Path(csv_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    if declared is not None and declared.sha256 is not None:
        if declared.sha256 != sha256:
            raise ValueError("CSV SHA-256 does not match the declared provenance.")

    if sha256 == _BUNDLED_GENERATED_SHA256:
        if declared is not None and declared.kind == "historical":
            raise ValueError("The bundled generated panel cannot be labeled historical.")
        return DataProvenance("synthetic", _BUNDLED_GENERATED_SOURCE, sha256)
    if declared is not None:
        return replace(declared, sha256=sha256)
    return DataProvenance("unknown", str(csv_path), sha256)

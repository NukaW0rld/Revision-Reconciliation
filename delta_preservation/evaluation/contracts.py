"""Contracts for immutable part-level ground truth fixtures."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TruthClassification = Literal["unchanged", "changed", "removed", "added"]


class GroundTruthContractError(ValueError):
    """Raised when a ground truth fixture cannot be resolved or validated."""


class GroundTruthCharacteristic(BaseModel):
    """Single canonical characteristic row from ``ground_truth.json``."""

    model_config = ConfigDict(extra="forbid")

    char_no: int | None = Field(
        None,
        ge=1,
        description="Characteristic number. Required for all non-added truth rows.",
    )
    classification: TruthClassification = Field(..., description="Expected reconciliation status.")
    requirement_revB: str | None = Field(
        None,
        description="Canonical Rev B requirement text. May be null only for removed rows.",
    )
    snippet_center_revA: tuple[float, float] | None = Field(
        None,
        description="Canonical Rev A snippet center in PDF points. May be null only for added rows.",
    )
    snippet_center_revB: tuple[float, float] | None = Field(
        None,
        description="Canonical Rev B snippet center in PDF points. May be null only for removed rows.",
    )

    @model_validator(mode="after")
    def _validate_status_specific_fields(self) -> "GroundTruthCharacteristic":
        errors: list[str] = []

        if self.classification != "added" and self.char_no is None:
            errors.append("char_no is required when classification is not 'added'")
        if self.classification != "removed" and self.requirement_revB is None:
            errors.append("requirement_revB is required unless classification is 'removed'")
        if self.classification != "added" and self.snippet_center_revA is None:
            errors.append("snippet_center_revA is required unless classification is 'added'")
        if self.classification != "removed" and self.snippet_center_revB is None:
            errors.append("snippet_center_revB is required unless classification is 'removed'")

        if errors:
            raise ValueError("; ".join(errors))

        return self


class GroundTruthPacket(BaseModel):
    """Validated top-level packet loaded from an immutable fixture file."""

    model_config = ConfigDict(extra="forbid")

    part_name: str = Field(..., description="Display part name recorded with the fixture.")
    general_notes: str = Field(..., description="Fixture-wide notes for evaluator context.")
    characteristics: list[GroundTruthCharacteristic] = Field(
        ...,
        description=(
            "Canonical characteristic expectations. Coverage mismatches are evaluation-time "
            "conditions and are not rejected here."
        ),
    )

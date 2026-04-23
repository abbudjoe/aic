"""Backend-neutral contracts for the AIC signal harness."""

from aic_signal_harness.artifacts import (
    HarnessIOError,
    read_json,
    sha256_file,
    write_json,
)
from aic_signal_harness.manifest import RunManifest, RunStatus
from aic_signal_harness.schemas import (
    ArtifactRef,
    BackendKind,
    ExperimentSpec,
    LeakageClass,
    PolicyBackendSpec,
    RuntimeBoundaryProof,
    RuntimeRole,
    SchemaValidationError,
    SimulatorKind,
    TrainingSourceKind,
    experiment_spec_from_json,
    experiment_spec_to_json,
    utc_now_iso,
)
from aic_signal_harness.scoring import ScoreReport, TrialScore, parse_scoring_yaml

__all__ = [
    "ArtifactRef",
    "BackendKind",
    "ExperimentSpec",
    "HarnessIOError",
    "LeakageClass",
    "PolicyBackendSpec",
    "RunManifest",
    "RunStatus",
    "ScoreReport",
    "RuntimeBoundaryProof",
    "RuntimeRole",
    "SchemaValidationError",
    "SimulatorKind",
    "TrainingSourceKind",
    "TrialScore",
    "experiment_spec_from_json",
    "experiment_spec_to_json",
    "parse_scoring_yaml",
    "read_json",
    "sha256_file",
    "utc_now_iso",
    "write_json",
]

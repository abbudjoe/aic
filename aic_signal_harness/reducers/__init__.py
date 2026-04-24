"""Typed reducer primitives for backend-neutral AIC harness artifacts."""

from aic_signal_harness.reducers.hdf5_dataset import (
    Hdf5DatasetReduction,
    reduce_hdf5_dataset,
    validate_hdf5_dataset,
)
from aic_signal_harness.reducers.ledger import build_ledger_entry
from aic_signal_harness.reducers.mcap_eval import (
    McapEvalReduction,
    analyze_mcap_eval_bundle,
    reduce_mcap_eval_bundle,
)
from aic_signal_harness.reducers.promotion import promote_ledger_entry
from aic_signal_harness.reducers.reward_failure import (
    RewardFailureReduction,
    derive_reward_failure_reports,
)
from aic_signal_harness.reducers.scoring_yaml import (
    ScoringYamlReduction,
    attach_scoring_yaml_reduction,
    reduce_scoring_yaml,
)

__all__ = [
    "Hdf5DatasetReduction",
    "McapEvalReduction",
    "RewardFailureReduction",
    "ScoringYamlReduction",
    "analyze_mcap_eval_bundle",
    "attach_scoring_yaml_reduction",
    "build_ledger_entry",
    "derive_reward_failure_reports",
    "promote_ledger_entry",
    "reduce_mcap_eval_bundle",
    "reduce_hdf5_dataset",
    "reduce_scoring_yaml",
    "validate_hdf5_dataset",
]

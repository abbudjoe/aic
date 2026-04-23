from pathlib import Path

import pytest

from aic_signal_harness import HarnessIOError, ScoreReport, TrialScore, parse_scoring_yaml


def test_parse_scoring_yaml_normalizes_total_and_trial_tiers(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 17.5
trial_1:
  tier_1:
    score: 1
    message: Model validation succeeded.
  tier_2:
    score: 2.5
    message: Scoring succeeded.
    categories:
      duration:
        score: 2.5
        message: Duration evidence is allowed as official detail.
  tier_3:
    score: 9
    message: Tier detail is allowed.
trial_2:
  tier_1:
    score: 1
  tier_2:
    score: 0
  tier_3:
    score: 4
""",
        encoding="utf-8",
    )

    report = parse_scoring_yaml(scoring)

    assert report.source == str(scoring)
    assert report.total == 17.5
    assert report.trial_count == 2
    assert report.trials["trial_1"] == TrialScore(
        total=12.5,
        tier_1=1.0,
        tier_2=2.5,
        tier_3=9.0,
    )
    assert report.trials["trial_2"] == TrialScore(
        total=5.0,
        tier_1=1.0,
        tier_2=0.0,
        tier_3=4.0,
    )
    assert report.parsed_at_utc is None


def test_parse_scoring_yaml_accepts_injected_parse_timestamp(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 1
trial_1:
  tier_1:
    score: 1
  tier_2:
    score: 0
  tier_3:
    score: 0
""",
        encoding="utf-8",
    )

    report = parse_scoring_yaml(scoring, parsed_at_utc="2026-04-23T00:00:00Z")

    assert report.parsed_at_utc == "2026-04-23T00:00:00Z"


def test_parse_scoring_yaml_is_deterministic_by_default(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 1
trial_1:
  tier_1:
    score: 1
  tier_2:
    score: 0
  tier_3:
    score: 0
""",
        encoding="utf-8",
    )

    assert parse_scoring_yaml(scoring).to_dict() == parse_scoring_yaml(scoring).to_dict()


def test_parse_scoring_yaml_missing_total_fails(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
trial_1:
  tier_1:
    score: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="missing required key: total"):
        parse_scoring_yaml(scoring)


def test_parse_scoring_yaml_rejects_empty_trial_set(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text("total: 0\n", encoding="utf-8")

    with pytest.raises(HarnessIOError, match="score.trials must contain at least one trial"):
        parse_scoring_yaml(scoring)


def test_parse_scoring_yaml_rejects_inconsistent_total(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 2
trial_1:
  tier_1:
    score: 1
  tier_2:
    score: 0
  tier_3:
    score: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="score.total does not match sum of trial totals"):
        parse_scoring_yaml(scoring)


def test_score_report_rejects_inconsistent_total() -> None:
    with pytest.raises(HarnessIOError, match="score.total does not match sum of trial totals"):
        ScoreReport(
            source="eval/scoring.yaml",
            total=10.0,
            trials={"trial_1": TrialScore(total=1.0, tier_1=1.0, tier_2=0.0, tier_3=0.0)},
        )

    payload = {
        "schema_version": 1,
        "source": "eval/scoring.yaml",
        "parsed_at_utc": None,
        "total": 10.0,
        "trial_count": 1,
        "trials": {
            "trial_1": {"total": 1.0, "tier_1": 1.0, "tier_2": 0.0, "tier_3": 0.0}
        },
    }
    with pytest.raises(HarnessIOError, match="score.total does not match sum of trial totals"):
        ScoreReport.from_dict(payload)


def test_trial_score_rejects_inconsistent_tier_total() -> None:
    with pytest.raises(HarnessIOError, match="trial.total does not match sum of tier scores"):
        TrialScore(total=2.0, tier_1=1.0, tier_2=0.0, tier_3=0.0)

    payload = {"total": 2.0, "tier_1": 1.0, "tier_2": 0.0, "tier_3": 0.0}
    with pytest.raises(HarnessIOError, match="trial.total does not match sum of tier scores"):
        TrialScore.from_dict(payload)


def test_score_report_rejects_empty_trials() -> None:
    with pytest.raises(HarnessIOError, match="score.trials must contain at least one trial"):
        ScoreReport(source="eval/scoring.yaml", total=0.0, trials={})

    payload = {
        "schema_version": 1,
        "source": "eval/scoring.yaml",
        "parsed_at_utc": None,
        "total": 0.0,
        "trial_count": 0,
        "trials": {},
    }
    with pytest.raises(HarnessIOError, match="score.trials must contain at least one trial"):
        ScoreReport.from_dict(payload)


def test_parse_scoring_yaml_rejects_numeric_trial_key(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 1
1:
  tier_1:
    score: 1
  tier_2:
    score: 0
  tier_3:
    score: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match=r"trial_\* strings"):
        parse_scoring_yaml(scoring)


def test_parse_scoring_yaml_rejects_missing_tier_structure(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 0
trial_1: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="trial 'trial_1' must contain tier_1"):
        parse_scoring_yaml(scoring)


def test_parse_scoring_yaml_rejects_non_mapping_trial_entry(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 1
trial_1: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="trial 'trial_1' must be a mapping"):
        parse_scoring_yaml(scoring)


def test_parse_scoring_yaml_rejects_present_non_mapping_tier(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 1
trial_1:
  tier_1: 1
  tier_2:
    score: 0
  tier_3:
    score: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="trial 'trial_1' tier_1 must be a mapping"):
        parse_scoring_yaml(scoring)


def test_parse_scoring_yaml_rejects_present_tier_without_score(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 1
trial_1:
  tier_1:
    note: missing score
  tier_2:
    score: 0
  tier_3:
    score: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="trial 'trial_1' tier_1 must contain score"):
        parse_scoring_yaml(scoring)


def test_parse_scoring_yaml_rejects_present_tier_with_non_finite_score(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 1
trial_1:
  tier_1:
    score: .inf
  tier_2:
    score: 0
  tier_3:
    score: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="trial 'trial_1' tier_1.score"):
        parse_scoring_yaml(scoring)


def test_score_report_round_trip_rejects_unknown_fields() -> None:
    report = ScoreReport(
        source="eval/scoring.yaml",
        parsed_at_utc="2026-04-23T00:00:00Z",
        total=1.0,
        trials={"trial_1": TrialScore(total=1.0, tier_1=1.0)},
    )

    assert ScoreReport.from_dict(report.to_dict()) == report

    payload = report.to_dict()
    payload["official_enough"] = True
    with pytest.raises(HarnessIOError, match="unknown fields"):
        ScoreReport.from_dict(payload)

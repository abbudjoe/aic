from pathlib import Path

from aic_lewm_policy.roadmap_tracking import validate_gate_review, validate_roadmap


def test_repository_roadmap_is_valid():
    repo_root = Path(__file__).resolve().parents[2]
    roadmap = repo_root / "aic_lewm_policy" / "experiments" / "roadmap.json"
    review = (
        repo_root
        / "aic_lewm_policy"
        / "experiments"
        / "gate_reviews"
        / "gate_0_harness_and_baseline.json"
    )

    assert validate_roadmap(roadmap)["ok"]
    assert validate_gate_review(review, roadmap_path=roadmap)["ok"]

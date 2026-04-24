from aic_signal_harness import (
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
)


class raises_schema_error:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            raise AssertionError("SchemaValidationError was not raised")
        if not issubclass(exc_type, SchemaValidationError):
            return False
        self.value = exc_value
        return True


def _assert_type_error(callback):
    try:
        callback()
    except TypeError:
        return
    raise AssertionError("TypeError was not raised")


def _set_mapping_item(mapping, key, value):
    mapping[key] = value


def _live_backend(**overrides):
    values = {
        "backend_kind": BackendKind.replay_servo,
        "name": "replay-servo-baseline",
        "runtime_role": RuntimeRole.live_policy,
        "training_sources": (TrainingSourceKind.official_demo,),
        "simulator_sources": (SimulatorKind.offline_replay,),
        "runtime_allowed": True,
        "leakage_class": LeakageClass.legal_policy_input,
        "runtime_boundary": RuntimeBoundaryProof(
            deterministic=True,
            uses_online_language_model_control=False,
            legal_observation_contract="official aic_model observations only",
        ),
        "description": "Replay a screened demonstration through the official runtime path.",
        "config": {"control_hz": 10},
        "provenance": {"producer": "pytest"},
    }
    values.update(overrides)
    return PolicyBackendSpec(**values)


def test_experiment_spec_round_trip_json():
    spec = ExperimentSpec(
        experiment_id="gate0-replay-smoke",
        hypothesis="A screened replay servo can validate the neutral experiment contract.",
        backend=_live_backend(),
        expected_artifacts=(
            ArtifactRef(
                kind="run_manifest",
                path="experiments/gate0/run_manifest.json",
                sha256="a" * 64,
                provenance={"producer": "pytest"},
            ),
        ),
        created_at_utc="2026-04-23T00:00:00Z",
        tags=("smoke", "contract"),
    )

    payload = experiment_spec_to_json(spec)
    decoded = experiment_spec_from_json(payload)

    assert decoded == spec
    assert decoded.to_dict()["backend"]["backend_kind"] == "replay_servo"
    assert decoded.to_dict()["expected_artifacts"][0]["sha256"] == "a" * 64


def test_live_policy_rejects_privileged_leakage():
    with raises_schema_error() as exc_info:
        _live_backend(leakage_class=LeakageClass.privileged_training_signal)

    assert "live_policy backend must use legal_policy_input leakage_class" in str(exc_info.value)


def test_runtime_boundary_proof_round_trip_with_policy_artifact():
    proof = RuntimeBoundaryProof(
        deterministic=True,
        uses_online_language_model_control=False,
        legal_observation_contract="official aic_model observations only",
        policy_artifact=ArtifactRef(
            kind="policy_checkpoint",
            uri="gs://bucket/policy.pt",
            sha256="b" * 64,
        ),
        notes="compiled before evaluation",
    )

    assert RuntimeBoundaryProof.from_dict(proof.to_dict()) == proof


def test_runtime_boundary_proof_requires_exact_booleans():
    with raises_schema_error() as exc_info:
        RuntimeBoundaryProof(
            deterministic=1,
            uses_online_language_model_control=False,
            legal_observation_contract="official aic_model observations only",
        )

    assert "runtime_boundary.deterministic must be a boolean" in str(exc_info.value)


def test_live_policy_requires_runtime_boundary_proof():
    with raises_schema_error() as exc_info:
        _live_backend(runtime_boundary=None)

    assert "live_policy backend must set runtime_boundary proof" in str(exc_info.value)


def test_live_policy_rejects_online_language_model_control():
    with raises_schema_error() as exc_info:
        _live_backend(
            runtime_boundary=RuntimeBoundaryProof(
                deterministic=True,
                uses_online_language_model_control=True,
                legal_observation_contract="official aic_model observations only",
            )
        )

    assert "live_policy runtime_boundary must not use online language-model control" in str(
        exc_info.value
    )


def test_high_risk_live_policy_requires_policy_artifact():
    for backend_kind in (
        BackendKind.isaac_rl,
        BackendKind.lerobot_act,
        BackendKind.lewm_world_model,
        BackendKind.open_vla,
        BackendKind.pi0,
    ):
        with raises_schema_error() as exc_info:
            _live_backend(backend_kind=backend_kind)

        assert (
            f"{backend_kind.value} live_policy must set runtime_boundary.policy_artifact"
            in str(exc_info.value)
        )


def test_offline_cosmos_role_rejected_as_live_runtime():
    with raises_schema_error() as exc_info:
        PolicyBackendSpec(
            backend_kind=BackendKind.cosmos_reason_critic,
            name="cosmos-reason-critic",
            runtime_role=RuntimeRole.offline_critic,
            training_sources=(TrainingSourceKind.model_annotation,),
            simulator_sources=(SimulatorKind.unknown,),
            runtime_allowed=True,
            leakage_class=LeakageClass.post_hoc_label,
            description="Offline VLM critic over reduced episode artifacts.",
        )

    message = str(exc_info.value)
    assert "offline_critic backend must not be runtime_allowed" in message
    assert "cosmos_reason_critic is an offline role" in message


def test_isaac_act_spec_accepted_as_non_privileged_live_policy():
    backend = PolicyBackendSpec(
        backend_kind=BackendKind.lerobot_act,
        name="isaac-act-policy",
        runtime_role=RuntimeRole.live_policy,
        training_sources=(TrainingSourceKind.official_demo, TrainingSourceKind.isaac_synthetic),
        simulator_sources=(SimulatorKind.isaac_lab,),
        runtime_allowed=True,
        leakage_class=LeakageClass.legal_policy_input,
        runtime_boundary=RuntimeBoundaryProof(
            deterministic=True,
            uses_online_language_model_control=False,
            legal_observation_contract="official aic_model observations only",
            policy_artifact=ArtifactRef(
                kind="policy_checkpoint",
                uri="s3://bucket/act_policy.pt",
            ),
        ),
        description="ACT imitation policy trained with Isaac Lab augmentation.",
        config={"architecture": "act"},
    )
    spec = ExperimentSpec(
        experiment_id="isaac-act-screen",
        hypothesis="ACT can be screened as a legal live policy after offline Isaac training.",
        backend=backend,
        expected_artifacts=(ArtifactRef(kind="policy_checkpoint", uri="s3://bucket/act_policy.pt"),),
        created_at_utc="2026-04-23T00:00:00Z",
        tags=("isaac", "act"),
    )

    assert spec.backend.backend_kind is BackendKind.lerobot_act
    assert spec.backend.runtime_allowed is True
    assert spec.backend.leakage_class is LeakageClass.legal_policy_input


def test_artifact_ref_rejects_malformed_path_uri_and_missing_location():
    with raises_schema_error() as exc_info:
        ArtifactRef(kind="run_manifest", path=5)

    message = str(exc_info.value)
    assert "artifact.path must be a nonempty string" in message
    assert "artifact must set path or uri" in message

    with raises_schema_error() as exc_info:
        ArtifactRef(kind="run_manifest", uri="")

    message = str(exc_info.value)
    assert "artifact.uri must be a nonempty string" in message
    assert "artifact must set path or uri" in message


def test_artifact_ref_rejects_bad_uri_shape():
    for uri in (
        "not a uri",
        "http://",
        "file://",
        "file:scoring.yaml",
        "file://host/tmp/a",
        "file:///tmp/%00x",
        "file:///tmp/%ZZ",
    ):
        with raises_schema_error() as exc_info:
            ArtifactRef(kind="policy_checkpoint", uri=uri)

        assert "artifact.uri" in str(exc_info.value)


def test_artifact_ref_rejects_bad_file_uri_identity_as_schema_error():
    for uri in ("file:///tmp/%00x", "file:///tmp/%ZZ"):
        with raises_schema_error() as exc_info:
            ArtifactRef(kind="policy_checkpoint", path="/tmp/x", uri=uri)

        assert "artifact.uri" in str(exc_info.value)


def test_artifact_ref_accepts_supported_uri_shapes():
    for uri in (
        "gs://bucket/path",
        "s3://bucket/key",
        "https://host/path",
        "file:///tmp/artifact.json",
    ):
        assert ArtifactRef(kind="policy_checkpoint", uri=uri).uri == uri


def test_artifact_ref_rejects_split_path_file_uri_identity(tmp_path):
    artifact_path = tmp_path / "artifact.json"
    other_path = tmp_path / "other.json"

    with raises_schema_error() as exc_info:
        ArtifactRef(kind="run_manifest", path=str(artifact_path), uri=other_path.as_uri())

    assert "path and file URI" in str(exc_info.value)


def test_artifact_ref_rejects_malformed_sha256_with_schema_error():
    for malformed_sha256 in (5, "not-a-digest"):
        with raises_schema_error() as exc_info:
            ArtifactRef(kind="run_manifest", path="run_manifest.json", sha256=malformed_sha256)

        assert "artifact.sha256 must be 64 hexadecimal characters" in str(exc_info.value)


def test_experiment_spec_rejects_malformed_expected_artifacts_json_field():
    payload = {
        "schema_version": 1,
        "experiment_id": "bad-artifacts",
        "hypothesis": "Malformed artifact containers fail closed.",
        "backend": _live_backend().to_dict(),
        "created_at_utc": "2026-04-23T00:00:00Z",
        "tags": ["schema"],
    }

    for expected_artifacts in (5, None):
        payload["expected_artifacts"] = expected_artifacts
        with raises_schema_error() as exc_info:
            ExperimentSpec.from_dict(payload)

        assert "expected_artifacts must be a list or tuple" in str(exc_info.value)


def test_experiment_spec_rejects_malformed_tags_json_field():
    payload = {
        "schema_version": 1,
        "experiment_id": "bad-tags",
        "hypothesis": "Malformed tag containers fail closed.",
        "backend": _live_backend().to_dict(),
        "expected_artifacts": [],
        "created_at_utc": "2026-04-23T00:00:00Z",
    }

    for tags in (5, None, "smoke"):
        payload["tags"] = tags
        with raises_schema_error() as exc_info:
            ExperimentSpec.from_dict(payload)

        assert "tags must be a list or tuple" in str(exc_info.value)


def test_policy_backend_from_dict_rejects_malformed_config_field():
    payload = _live_backend().to_dict()

    for config in ([], 0):
        payload["config"] = config
        with raises_schema_error() as exc_info:
            PolicyBackendSpec.from_dict(payload)

        assert "backend.config must be a mapping" in str(exc_info.value)


def test_policy_backend_rejects_policy_artifact_config_aliases():
    for key in (
        "checkpoint",
        "checkpoint_file",
        "checkpoint_uri",
        "model_checkpoint",
        "policy_artifact",
        "policy_artifact_uri",
        "policy_path",
        "policy_uri",
        "weights_path",
        "weights_uri",
    ):
        with raises_schema_error() as exc_info:
            _live_backend(config={key: "s3://bucket/other-policy.pt"})

        assert "backend.config must not define policy artifact keys" in str(exc_info.value)


def test_policy_backend_allows_non_artifact_tuning_config_keys():
    backend = _live_backend(
        config={
            "gradient_checkpointing": True,
            "loss_weights": {"score": 1.0},
        }
    )

    assert backend.config["gradient_checkpointing"] is True
    assert backend.config["loss_weights"] == {"score": 1.0}


def test_validated_mappings_are_deeply_immutable():
    backend = _live_backend(
        config={
            "gradient_checkpointing": True,
            "loss_weights": {"score": 1.0},
        },
        provenance={"screen": {"owner": "pytest"}},
    )
    artifact = ArtifactRef(
        kind="run_manifest",
        path="run_manifest.json",
        provenance={"nested": {"owner": "pytest"}},
    )

    _assert_type_error(lambda: _set_mapping_item(backend.config, "policy_path", "s3://bucket/policy.pt"))
    _assert_type_error(lambda: _set_mapping_item(backend.config["loss_weights"], "score", 2.0))
    _assert_type_error(lambda: _set_mapping_item(backend.provenance["screen"], "owner", "mutated"))
    _assert_type_error(lambda: _set_mapping_item(artifact.provenance["nested"], "owner", "mutated"))

    assert backend.to_dict()["config"]["loss_weights"] == {"score": 1.0}
    assert artifact.to_dict()["provenance"]["nested"] == {"owner": "pytest"}


def test_policy_backend_rejects_nested_policy_artifact_config_aliases():
    for config in (
        {"paths": {"policy_path": "s3://bucket/other-policy.pt"}},
        {"layers": [{"weights_uri": "s3://bucket/weights.safetensors"}]},
        {"policy": {"path": "s3://bucket/policy.pt"}},
        {"model": {"uri": "s3://bucket/model.pt"}},
        {"artifacts": {"policy": "s3://bucket/policy.pt"}},
        {"weights": "s3://bucket/weights.pt"},
    ):
        with raises_schema_error() as exc_info:
            _live_backend(config=config)

        assert "backend.config must not define policy artifact keys" in str(exc_info.value)


def test_policy_backend_allows_non_artifact_split_config_context():
    backend = _live_backend(
        config={
            "model": {"name": "resnet18", "layers": 18},
            "policy": {"temperature": 0.0},
            "weights": [0.2, 0.8],
        }
    )

    assert backend.config["model"]["name"] == "resnet18"
    assert backend.config["policy"]["temperature"] == 0.0
    assert backend.config["weights"] == (0.2, 0.8)


def test_from_dict_rejects_unknown_schema_fields():
    backend_payload = _live_backend().to_dict()
    backend_payload["policy_path"] = "s3://bucket/hidden-policy.pt"
    with raises_schema_error() as exc_info:
        PolicyBackendSpec.from_dict(backend_payload)

    assert "backend has unknown fields" in str(exc_info.value)

    spec_payload = {
        "schema_version": 1,
        "experiment_id": "unknown-field",
        "hypothesis": "Unknown fields fail closed.",
        "backend": _live_backend().to_dict(),
        "expected_artifacts": [],
        "created_at_utc": "2026-04-23T00:00:00Z",
        "tags": [],
        "policy_path": "s3://bucket/hidden-policy.pt",
    }
    with raises_schema_error() as exc_info:
        ExperimentSpec.from_dict(spec_payload)

    assert "experiment spec has unknown fields" in str(exc_info.value)


def test_policy_backend_rejects_non_string_config_keys():
    with raises_schema_error() as exc_info:
        _live_backend(config={5: "not-json-object-shaped"})

    assert "backend.config keys must be nonempty strings" in str(exc_info.value)


def test_provenance_rejects_non_string_nested_json_keys():
    with raises_schema_error() as exc_info:
        ArtifactRef(kind="run_manifest", path="run_manifest.json", provenance={"nested": {5: "bad"}})

    assert "artifact.provenance.nested keys must be nonempty strings" in str(exc_info.value)

    with raises_schema_error() as exc_info:
        _live_backend(provenance={"nested": {5: "bad"}})

    assert "backend.provenance.nested keys must be nonempty strings" in str(exc_info.value)


def test_policy_backend_from_dict_rejects_scalar_source_fields():
    for field_name, value in (
        ("training_sources", "official_demo"),
        ("simulator_sources", "offline_replay"),
    ):
        payload = _live_backend().to_dict()
        payload[field_name] = value

        with raises_schema_error() as exc_info:
            PolicyBackendSpec.from_dict(payload)

        assert f"{field_name} must be a list or tuple" in str(exc_info.value)


def test_policy_backend_constructor_rejects_scalar_source_fields():
    for field_name, value in (
        ("training_sources", "official_demo"),
        ("simulator_sources", "offline_replay"),
    ):
        with raises_schema_error() as exc_info:
            _live_backend(**{field_name: value})

        assert f"{field_name} must be a list or tuple" in str(exc_info.value)


def test_custom_backend_requires_nonempty_json_provenance():
    with raises_schema_error() as exc_info:
        _live_backend(backend_kind=BackendKind.custom, provenance={})

    assert "custom backend must set nonempty provenance" in str(exc_info.value)

    with raises_schema_error() as exc_info:
        _live_backend(backend_kind=BackendKind.custom, provenance=5)

    assert "backend.provenance must be a mapping" in str(exc_info.value)

    malformed_payload = _live_backend(backend_kind=BackendKind.custom).to_dict()
    malformed_payload["provenance"] = []
    with raises_schema_error() as exc_info:
        PolicyBackendSpec.from_dict(malformed_payload)

    assert "backend.provenance must be a mapping" in str(exc_info.value)

    backend = _live_backend(
        backend_kind=BackendKind.custom,
        provenance={"owner": "pytest", "screen": "custom-contract"},
    )

    decoded = PolicyBackendSpec.from_dict(backend.to_dict())
    assert decoded.provenance == {"owner": "pytest", "screen": "custom-contract"}

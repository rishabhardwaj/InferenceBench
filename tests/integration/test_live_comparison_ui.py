from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest


API_KEY = "doo_v1_streamlit-live-secret"


def test_paid_form_is_the_only_trigger_and_opens_fresh_saved_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_fixture_live_environment(tmp_path, monkeypatch)
    real_async_client = httpx.AsyncClient
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        model_id = payload["model"]
        calls.append(model_id)
        if calls.count(model_id) == 1 and len(set(calls)) == 1:
            return httpx.Response(
                500,
                headers={"x-request-id": "ui-redaction-check"},
                json={"error": f"Bearer {API_KEY}"},
            )
        return httpx.Response(
            200,
            headers={"x-request-id": f"ui-request-{len(calls)}"},
            json={
                "id": f"ui-response-{len(calls)}",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "bug"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "total_tokens": 101,
                },
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_async_client(
            transport=httpx.MockTransport(handler)
        ),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("live UI startup must not call GitHub")
        ),
    )

    app = AppTest.from_file(Path(__file__).resolve().parents[2] / "app.py").run(
        timeout=20
    )
    assert not app.exception
    assert calls == []

    app.radio[0].set_value("Run Comparison View").run(timeout=20)
    assert not app.exception
    assert calls == []
    assert len(app.selectbox) == 2
    assert app.selectbox[0].label == "Model A"
    assert app.selectbox[1].label == "Model B"
    assert len(app.selectbox[0].options) == 25
    assert app.selectbox[0].options == app.selectbox[1].options
    assert "openai-gpt-4o" not in app.selectbox[0].options
    assert len(app.number_input) == 1
    assert app.number_input[0].value == 2

    model_a = app.selectbox[0].options[2]
    model_b = app.selectbox[1].options[3]
    app.selectbox[0].set_value(model_a)
    app.selectbox[1].set_value(model_a)
    app.run(timeout=20)
    same_model_start = next(
        button
        for button in app.button
        if button.label == "Start new comparison — paid action"
    )
    assert same_model_start.disabled is True
    assert calls == []

    app.selectbox[0].set_value(model_a)
    app.selectbox[1].set_value(model_b)
    app.number_input[0].set_value(2)
    app.run(timeout=20)
    assert calls == []

    start = next(
        button
        for button in app.button
        if button.label == "Start new comparison — paid action"
    )
    assert start.disabled is False
    start.click().run(timeout=30)

    assert not app.exception
    assert calls == [model_a] * 7 + [model_b] * 7
    rendered_text = "\n".join(
        [
            *(element.value for element in app.markdown),
            *(element.value for element in app.caption),
            *(element.value for element in app.info),
            *(element.value for element in app.success),
            *(element.value for element in app.warning),
            *(element.value for element in app.subheader),
        ]
    )
    assert "2 × 7 = 14" in rendered_text
    assert "Persisted `7/7`" in rendered_text
    assert "known cost" in rendered_text
    assert "Both fresh independent runs completed" in rendered_text
    assert "Live result — fresh persisted comparison" in rendered_text
    assert model_a in rendered_text
    assert model_b in rendered_text

    database_path = tmp_path / "ui-live.sqlite3"
    assert API_KEY.encode() not in database_path.read_bytes()
    with sqlite3.connect(database_path) as connection:
        manifests = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT manifest_json FROM run_manifests"
            ).fetchall()
        ]
    live_runs = [item for item in manifests if item["run_type"] == "model_evaluation"]
    assert len(live_runs) == 2
    assert {item["model_id"] for item in live_runs} == {model_a, model_b}
    assert all(item["status"] == "complete" for item in live_runs)
    assert all(item["concurrency"] == 2 for item in live_runs)
    assert len({item["run_id"] for item in live_runs}) == 2

    # The completed review remains read-only: changing one of its result
    # filters reopens persisted evidence and cannot silently start another run.
    provider_call_count = len(calls)
    unscored_issue_filter = app.selectbox[-1]
    unscored_issue_filter.set_value(unscored_issue_filter.options[-1]).run(timeout=20)
    assert not app.exception
    assert len(calls) == provider_call_count


def test_masked_session_key_is_disclosed_server_side_and_can_be_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_fixture_live_environment(tmp_path, monkeypatch)
    monkeypatch.delenv("DO_INFERENCE_API_KEY")
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("entering or clearing a session key must not infer")
        ),
    )

    app = AppTest.from_file(Path(__file__).resolve().parents[2] / "app.py").run(
        timeout=20
    )
    app.radio[0].set_value("Run Comparison View").run(timeout=20)

    assert not app.exception
    assert len(app.text_input) == 1
    assert app.text_input[0].label == "DigitalOcean API key (active session only)"
    # Streamlit's testing wrapper calls every st.text_input a "text_input";
    # the protobuf field is what distinguishes a masked password input.
    assert app.text_input[0].proto.type == 1
    disclosure = "\n".join(element.value for element in app.caption)
    assert "not browser-only" in disclosure
    assert "HTTPS" in disclosure
    start = next(
        button
        for button in app.button
        if button.label == "Start new comparison — paid action"
    )
    assert start.disabled is True

    app.text_input[0].set_value(API_KEY).run(timeout=20)
    start = next(
        button
        for button in app.button
        if button.label == "Start new comparison — paid action"
    )
    assert start.disabled is False
    assert API_KEY.encode() not in (tmp_path / "ui-live.sqlite3").read_bytes()

    clear = next(
        button for button in app.button if button.label == "Clear session API key"
    )
    clear.click().run(timeout=20)
    assert not app.exception
    assert app.text_input[0].value == ""
    start = next(
        button
        for button in app.button
        if button.label == "Start new comparison — paid action"
    )
    assert start.disabled is True


def _configure_fixture_live_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_version = "fixture-corpus-v1"
    corpus_directory = corpus_root / corpus_version
    shutil.copytree(Path("artifacts/fixtures/corpus/v1"), corpus_directory)
    corpus_manifest = json.loads(
        (corpus_directory / "manifest.json").read_text(encoding="utf-8")
    )
    (corpus_root / "default.json").write_text(
        json.dumps(
            {
                "schema_version": "active_corpus.v1",
                "corpus_version": corpus_version,
                "manifest_file": f"{corpus_version}/manifest.json",
                "artifact_sha256": corpus_manifest["artifact_sha256"],
            }
        ),
        encoding="utf-8",
    )

    ground_truth_root = tmp_path / "ground-truth"
    evaluation_version = "fixture-ground-truth-v1"
    shutil.copytree(
        Path("artifacts/fixtures/ground_truth/v1"),
        ground_truth_root / evaluation_version,
    )

    contract_directory = tmp_path / "frozen-contract"
    shutil.copytree(Path("artifacts/prompts/development-v1"), contract_directory)
    manifest_path = contract_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "contract_version": "shared-inference-contract-ui-frozen-v1",
            "contract_status": "frozen",
            "prompt_version": "zero-shot-ui-frozen-v1",
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("INFERENCEBENCH_DB_PATH", str(tmp_path / "ui-live.sqlite3"))
    monkeypatch.setenv("INFERENCEBENCH_CORPUS_ROOT", str(corpus_root))
    monkeypatch.setenv("INFERENCEBENCH_GROUND_TRUTH_ROOT", str(ground_truth_root))
    monkeypatch.setenv("INFERENCEBENCH_EVALUATION_VERSION", evaluation_version)
    monkeypatch.setenv("INFERENCEBENCH_CONTRACT_PATH", str(contract_directory))
    monkeypatch.setenv("INFERENCEBENCH_DEFAULT_CONCURRENCY", "2")
    monkeypatch.setenv("INFERENCEBENCH_SHARED_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("DO_INFERENCE_API_KEY", API_KEY)

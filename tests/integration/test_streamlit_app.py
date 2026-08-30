from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest


def test_app_opens_saved_comparison_without_provider_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INFERENCEBENCH_DB_PATH", str(tmp_path / "ui.sqlite3"))
    monkeypatch.delenv("DO_INFERENCE_API_KEY", raising=False)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("application startup must not call GitHub")
        ),
    )
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("saved review must not create an inference client")
        ),
    )

    app_path = Path(__file__).resolve().parents[2] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "InferenceBench"
    assert any("Frozen Corpus" in caption.value for caption in app.caption)
    assert any("Frozen model and pricing evidence" in caption.value for caption in app.caption)
    assert any("Saved evidence" in caption.value for caption in app.caption)
    assert app.radio[0].label == "Application view"
    assert app.radio[0].value == "Saved Comparison Review"
    assert len(app.selectbox) == 4
    assert app.selectbox[0].label == "Evidence stratum"
    assert app.selectbox[0].value == "Primary Scored Holdout"
    assert app.selectbox[0].options == ["Primary Scored Holdout"]
    assert app.selectbox[1].label == "Model prediction relationship"
    assert app.selectbox[2].label == "Issue detail"
    assert app.selectbox[3].label == "Unscored issue detail"
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Known Total Calculated Cost"] == "$0.00001725"
    assert metrics["Strict Agreement Rate"] == "16.7%"
    assert metrics["Both-Valid Agreement Rate"] == "50.0%"
    assert metrics["Label agreement"] == "1"
    assert metrics["Label disagreement"] == "1"
    assert metrics["One-sided failure"] == "2"
    assert metrics["Joint failure"] == "2"
    assert len(app.multiselect) == 12
    assert {control.label for control in app.multiselect} == {
        "Ground Truth Label",
        "Model A prediction",
        "Model B prediction",
        "Model A Scored Outcome",
        "Model B Scored Outcome",
        "Ground Truth provenance",
        "Sampling stratum",
        "Unscored Model A prediction",
        "Unscored Model B prediction",
        "Unscored Pair Outcome",
        "Unscored Model A result state",
        "Unscored Model B result state",
    }
    assert len(app.dataframe[1].value) == 6
    assert len(app.dataframe[2].value) == 6
    assert "No Valid Prediction" in app.dataframe[3].value.columns
    assert "No Valid Prediction" in app.dataframe[4].value.columns
    assert app.dataframe[5].value.iloc[0]["Pair outcome"] == "label_disagreement"
    assert len(app.dataframe[6].value) == 7
    assert len(app.dataframe[7].value) == 7
    assert app.dataframe[6].value.iloc[-1].to_dict() == {
        "Suggestion": "No Valid Prediction",
        "Count": 3,
        "Expected": 6,
        "Rate": "50.0%",
    }
    assert app.dataframe[7].value.iloc[-1].to_dict() == {
        "Suggestion": "No Valid Prediction",
        "Count": 3,
        "Expected": 6,
        "Rate": "50.0%",
    }
    assert app.dataframe[8].value.iloc[0]["Pair outcome"] == "label_disagreement"
    assert "Ground Truth" not in app.dataframe[8].value.columns
    rendered_text = "\n".join(
        [
            *(element.value for element in app.markdown),
            *(element.value for element in app.caption),
            *(element.value for element in app.subheader),
        ]
    )
    assert "openai-gpt-oss-20b" in rendered_text
    assert "openai-gpt-oss-120b" in rendered_text
    assert "fixture-corpus-v1" in rendered_text
    assert "fixture-shared-contract-v0" in rendered_text
    assert "fixture-v2-run-model-a" in rendered_text
    assert "fixture-v2-run-model-b" in rendered_text
    assert "doctl-2026-08-30" in rendered_text
    assert "2f1db01f91a3ccc21c2ac0c3b10dc9720dde450d4643e0a4de7ece80a7e3b711" in rendered_text
    assert "eligible-candidates-2026-08-29" in rendered_text
    assert "do-serverless-pricing-2026-08-30" in rendered_text
    assert "calculated-request-cost-v1" in rendered_text
    assert "linear-percentile-v1" in rendered_text
    assert "partial lower bound" in rendered_text
    assert "undefined (0 correct classifications)" in rendered_text
    assert "usable `4/7`" in rendered_text
    assert "Sustained Request Throughput" in rendered_text
    assert "Queue Wait (separate from Request Latency)" in rendered_text
    assert "Scored View" in rendered_text
    assert "Coverage 1/6 labels" in rendered_text
    assert "Showing 1/1 scored issues" in rendered_text
    assert "Unscored View" in rendered_text
    assert "1/6 expected unscored issues" in rendered_text
    assert "1/2 issues with two valid labels" in rendered_text
    assert "Showing 6/6 unscored issues" in rendered_text

    app.selectbox[3].set_value(7).run(timeout=20)
    failure_text = "\n".join(
        [
            *(element.value for element in app.markdown),
            *(element.value for element in app.caption),
        ]
    )
    assert "**Unscored Pair Outcome:** `joint_failure`" in failure_text
    assert failure_text.count("**Typed terminal error**") >= 2
    assert failure_text.count("**Provider outcome:** `timeout`") >= 2

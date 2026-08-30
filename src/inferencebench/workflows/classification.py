from __future__ import annotations

import httpx

from inferencebench.domain import Issue
from inferencebench.inference.contract import (
    load_shared_inference_contract,
    prepare_classification_request,
)
from inferencebench.inference.digitalocean import DigitalOceanChatCompletionsAdapter
from inferencebench.inference.domain import (
    SharedInferenceContract,
    SingleIssueClassificationResult,
)


async def classify_one_frozen_issue(
    issue: Issue,
    *,
    model_id: str,
    api_key: str,
    timeout_seconds: float,
    client: httpx.AsyncClient,
    contract: SharedInferenceContract | None = None,
) -> SingleIssueClassificationResult:
    active_contract = contract or load_shared_inference_contract()
    request = prepare_classification_request(issue, model_id, active_contract)
    adapter = DigitalOceanChatCompletionsAdapter(client)
    return await adapter.classify(
        request,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )

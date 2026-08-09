from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = (
    PROJECT_ROOT / "configs/integrations/feishu_lifetwin_workflow_v1.json"
)


def test_feishu_workflow_keeps_numeric_prediction_in_model_tool() -> None:
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert workflow["status"] == (
        "deployment_blueprint_requires_enterprise_tenant_credentials"
    )
    assert workflow["data_boundary"][
        "raw_bms_and_rpt_leave_enterprise_domain"
    ] is False
    tool_ids = {tool["tool_id"] for tool in workflow["aily_tools"]}
    assert "run_lifetwin_prediction" in tool_ids
    assert any(
        "Never calculate or guess SOH" in rule
        for rule in workflow["agent_guardrails"]
    )
    assert any(
        "Never translate a refusal" in rule
        for rule in workflow["agent_guardrails"]
    )


def test_feishu_workflow_has_auditable_relational_keys() -> None:
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    tables = {table["table_id"]: table for table in workflow["base_tables"]}
    assert set(tables) == {
        "cell_registry",
        "model_registry",
        "prediction_tasks",
        "prediction_evidence",
        "approval_and_truth",
    }
    assert "prediction_sha256" in tables["prediction_evidence"]["fields"]
    assert "data_manifest_sha256" in tables["prediction_tasks"]["fields"]
    assert "score_sha256" in tables["approval_and_truth"]["fields"]

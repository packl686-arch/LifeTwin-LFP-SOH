from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
CONSOLE = ROOT / "docs" / "v5-console" / "index.html"


def test_v5_console_publishes_exact_summary_and_boundaries() -> None:
    html = CONSOLE.read_text(encoding="utf-8")
    assert "0.2082 pp" in html
    assert "93.10%" in html
    assert "1.4155 pp" in html
    assert "[-0.1503, -0.0175] pp" in html
    assert "不含海辰内部数据" in html
    assert "不构成 15-25 年准确率承诺" in html
    assert "完整 GP 在线 landmark 门槛已执行但未通过" in html
    assert "公开开发回放 · 非生产系统" in html


def test_v5_console_links_and_release_registration_are_valid() -> None:
    html = CONSOLE.read_text(encoding="utf-8")
    assert "../assets/v5_fastcharge_development_results.png" in html
    assert (
        ROOT / "docs" / "assets" / "v5_fastcharge_development_results.png"
    ).is_file()
    assert "../assets/v5_dynamic_landmark_audit.png" in html
    assert (ROOT / "docs" / "assets" / "v5_dynamic_landmark_audit.png").is_file()
    assert "../../showcase/evidence_v5/README.md" in html
    assert (ROOT / "showcase" / "evidence_v5" / "README.md").is_file()
    assert "../../reports/fastcharge_v5_dynamic_landmark_audit_2026-08-09.md" in html
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    assert "docs/v5-console/index.html" in manifest["unfrozen_tracked_allowlist"]
    assert "tests/test_v5_console.py" in manifest["unfrozen_tracked_allowlist"]


def test_v5_console_contains_all_interactive_workflow_contracts() -> None:
    html = CONSOLE.read_text(encoding="utf-8")
    for prefix in (20, 40, 60, 100):
        assert f'data-prefix="{prefix}"' in html
    for tool in (
        "validate_lifetwin_input",
        "run_lifetwin_prediction",
        "register_lifetwin_result",
        "score_lifetwin_prediction",
    ):
        assert tool in html
    assert 'id="performance-chart"' in html
    assert 'id="workflow-next"' in html
    assert 'id="quality"' in html


def test_root_readme_exposes_current_v5_review_path_and_claim_boundary() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## 2026 参赛当前版本" in readme
    assert "v5-console/" in readme
    assert "fastcharge_v5_dynamic_landmark_audit_2026-08-09.md" in readme
    assert "仓库不包含海辰内部测量" in readme
    assert "不是海辰产品验证、日历老化确认或 15 至 25 年准确率证明" in readme


def test_competition_evidence_pack_uses_current_v5_results() -> None:
    pack = (ROOT / "docs" / "competition_final_evidence_pack_cn.md").read_text(
        encoding="utf-8"
    )
    assert "V5 参考条件化残差" in pack
    assert "总体 MAE 降低 **27.3%**" in pack
    assert "覆盖率为 **93.10%**" in pack
    assert "不启用额外在线残差分支" in pack
    assert "v5-console/" in pack

    guide = (
        ROOT / "docs" / "independent_validation_execution_2026_08_cn.md"
    ).read_text(encoding="utf-8")
    assert "FastCharge V5 是当前最强" in guide
    assert "仍不是独立长期日历老化或海辰产品验证" in guide


def test_competition_deck_uses_current_v5_story() -> None:
    deck = ROOT / "docs" / "LifeTwin_competition_defense_cn.pptx"
    namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    visible_text: list[str] = []
    notes_text: list[str] = []
    with ZipFile(deck) as archive:
        for name in archive.namelist():
            if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
                root = ElementTree.fromstring(archive.read(name))
                visible_text.extend(
                    node.text or "" for node in root.findall(".//a:t", namespace)
                )
            if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml"):
                root = ElementTree.fromstring(archive.read(name))
                notes_text.extend(
                    node.text or "" for node in root.findall(".//a:t", namespace)
                )

    joined = "\n".join(visible_text)
    assert "公开开发：V5相对V2降低27.3%" in joined
    assert "0.208 pp" in joined
    assert "93.10%" in joined
    assert "在线GP：不激活" in joined
    assert "飞书AI + 私有盲测：三道防火墙" in joined
    assert "[Sources]" in "\n".join(notes_text)

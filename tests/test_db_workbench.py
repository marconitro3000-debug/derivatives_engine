from __future__ import annotations

from db.database import get_workflow, init_db, insert_pricing_run, insert_workflow


def test_pricing_run_and_workflow_roundtrip(tmp_path):
    db_file = tmp_path / "test.sqlite"
    init_db(db_file)
    run_id = insert_pricing_run("phoenix_autocall", {"spot": 100}, {"fair_value": 101}, path=db_file)
    workflow_id = insert_workflow("demo", "phoenix_autocall", {"nodes": [], "edges": []}, path=db_file)
    workflow = get_workflow(workflow_id, path=db_file)
    assert run_id > 0
    assert workflow is not None
    assert workflow["workflow"]["nodes"] == []

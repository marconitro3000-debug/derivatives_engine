"""Repository-style wrappers over the SQLite helper functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from db.database import get_workflow, insert_pricing_run, insert_workflow


class PricingRunRepository:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = path

    def insert(self, product_type: str, request: dict[str, Any], response: dict[str, Any]) -> int:
        return insert_pricing_run(product_type, request, response, self.path)


class WorkflowRepository:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = path

    def insert(self, name: str, product_type: str, workflow: dict[str, Any]) -> int:
        return insert_workflow(name, product_type, workflow, self.path)

    def get(self, workflow_id: int) -> dict[str, Any] | None:
        return get_workflow(workflow_id, self.path)

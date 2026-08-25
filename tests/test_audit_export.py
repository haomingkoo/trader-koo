from __future__ import annotations

import csv
import io
from pathlib import Path

from trader_koo.audit.export import export_logs_to_local


class _AuditLogQuery:
    def query_logs(self, **_filters):
        return [{"event_type": "login", "details": {"success": True}}]


def test_export_logs_to_local_writes_csv(tmp_path: Path):
    result = export_logs_to_local(
        _AuditLogQuery(),  # type: ignore[arg-type]
        export_format="csv",
        export_dir=tmp_path,
    )

    output = Path(result["location"])
    assert output.parent == tmp_path
    rows = list(csv.DictReader(io.StringIO(output.read_text())))
    assert rows == [{"event_type": "login", "details": '{"success": true}'}]

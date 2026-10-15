import json

import pytest

from runtime.decisions import ProtocolError, decision_schema, parse_tool_payload


def test_parse_tool_payload_reports_all_top_level_schema_errors():
    with pytest.raises(ProtocolError) as error:
        parse_tool_payload(json.dumps({"unexpected": True}))

    message = str(error.value)
    assert message.splitlines() == [
        "1. missing top-level key(s): arguments, tool",
        "2. unexpected top-level key(s): unexpected",
    ]


def test_strict_probe_schema_removes_batch_skip_tool():
    assert "request_next_batch" not in decision_schema(enable_probe_skip=False)[
        "properties"
    ]["tool"]["enum"]
    with pytest.raises(ProtocolError, match="tool does not exist"):
        parse_tool_payload(
            '{"tool":"request_next_batch","arguments":{}}',
            enable_probe_skip=False,
        )

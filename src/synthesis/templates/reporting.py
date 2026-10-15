"""Read-only human reporting for accepted canonical template rationales."""
from __future__ import annotations

from typing import Any


def rationale_rows(pool: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain, domain_value in pool["domains"].items():
        for template_id, template in domain_value["templates"].items():
            for group in template["dynamics"]["replacements"].values():
                for transition_id, transition in group["transitions"].items():
                    rows.append({
                        "domain": domain, "template_id": template_id,
                        "dynamic_type": "field_replacement", "event_id": transition_id,
                        "subject": f"{transition['outgoing_field']} -> {transition['incoming_field']}",
                        "trigger": transition["trigger"],
                        "before": transition["outgoing_retirement_reason"],
                        "after": transition["incoming_authority_reason"],
                        "scope": group["effective_scope"],
                        "duration": transition["expected_duration"],
                        "ending_condition": transition["ending_condition"],
                        "rationale": group["rationale"],
                    })
            for field_id, event in template["dynamics"]["direction_reversals"].items():
                rows.append({
                    "domain": domain, "template_id": template_id,
                    "dynamic_type": "direction_reversal", "event_id": field_id,
                    "subject": field_id, "trigger": event["trigger"],
                    "before": event["semantics_before"], "after": event["semantics_after"],
                    "scope": None, "duration": None,
                    "ending_condition": None,
                    "rationale": event["reversal_reason"],
                })
    return rows


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# AdaExploreBench Template Dynamic Rationales", ""]
    for row in rows:
        lines.extend([
            f"## {row['domain']} / {row['template_id']} / {row['dynamic_type']} / {row['event_id']}", "",
            f"- Subject: {row['subject']}", f"- Trigger: {row['trigger']}",
            f"- Before: {row['before']}", f"- After: {row['after']}",
        ])
        if row.get("scope"):
            lines.append(f"- Scope: {row['scope']}")
        if row.get("duration"):
            lines.append(f"- Duration: {row['duration']}")
        if row.get("ending_condition"):
            lines.append(f"- Ending condition: {row['ending_condition']}")
        lines.extend(["", row["rationale"], ""])
    return "\n".join(lines).rstrip() + "\n"

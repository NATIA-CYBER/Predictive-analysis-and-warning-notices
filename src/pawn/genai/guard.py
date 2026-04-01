from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable

MAX_FIELD_LEN = 500

ALLOWLIST = {
    "dept",
    "weekly_ts",
    "headcount",
    "leavers",
    "attrition_rate",
    "top_drivers",
    "risk_score",
    "threshold",
    "notes",
    "warning",
}


def _clip_text(value: str, limit: int = MAX_FIELD_LEN) -> str:
    value = value.replace("\r", " ").replace("\n", " ").strip()
    return value[:limit]


def sanitize(alert: Dict[str, Any], allowlist: Iterable[str] = ALLOWLIST) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}

    for key in allowlist:
        if key not in alert:
            continue

        value = alert[key]

        if isinstance(value, str):
            safe[key] = _clip_text(value)
        elif isinstance(value, (int, float, bool)):
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            trimmed = []
            for item in value[:20]:
                if isinstance(item, str):
                    trimmed.append(_clip_text(item))
                else:
                    trimmed.append(item)
            safe[key] = trimmed
        elif isinstance(value, dict):
            trimmed_dict = {}
            for sub_key, sub_val in list(value.items())[:20]:
                if isinstance(sub_val, str):
                    trimmed_dict[sub_key] = _clip_text(sub_val)
                else:
                    trimmed_dict[sub_key] = sub_val
            safe[key] = trimmed_dict
        else:
            safe[key] = _clip_text(str(value))

    return safe


def _fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{100 * float(value):.1f}%"
    return "n/a"


def _fmt_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "n/a"


def _normalize_driver_text(driver: str) -> str:
    d = driver.strip().lower()

    replacements = {
        "v1_overtime_sum": "overtime pressure",
        "v2_overload_satisfaction_sum": "high workload with low satisfaction",
        "v3_stagnation_sum": "career stagnation",
        "v4_post_accident_overwork_sum": "post-accident overwork pattern",
        "violation_density": "elevated violation density",
        "average_montly_hours_mean": "high average monthly hours",
        "time_spend_company_mean": "long tenure concentration",
        "satisfaction_level_mean": "lower average satisfaction",
        "attrition_rate": "elevated attrition rate",
        "leavers": "higher number of leavers",
    }

    return replacements.get(d, d.replace("_", " "))


def _extract_top_drivers(alert: Dict[str, Any]) -> list[str]:
    drivers = alert.get("top_drivers", [])
    if not isinstance(drivers, (list, tuple)):
        return []

    cleaned: list[str] = []
    for item in drivers[:5]:
        if isinstance(item, str):
            cleaned.append(_normalize_driver_text(item))
        else:
            cleaned.append(str(item))
    return cleaned


def summarize_offline(alert: Dict[str, Any]) -> str:
    dept = alert.get("dept", "unknown")
    weekly_ts = alert.get("weekly_ts", "n/a")
    risk_score = alert.get("risk_score")
    threshold = alert.get("threshold")
    attrition_rate = alert.get("attrition_rate")
    leavers = alert.get("leavers")
    headcount = alert.get("headcount")
    warning = bool(alert.get("warning", False))
    drivers = _extract_top_drivers(alert)
    notes = alert.get("notes")

    lines: list[str] = []
    lines.append(f"Department {dept} for week {weekly_ts} was scored by the PAWN warning layer.")
    lines.append(
        f"Risk score is {_fmt_score(risk_score)} against threshold {_fmt_score(threshold)}. "
        f"Warning status: {'triggered' if warning else 'not triggered'}."
    )

    context_parts: list[str] = []
    if headcount is not None:
        context_parts.append(f"headcount {headcount}")
    if leavers is not None:
        context_parts.append(f"leavers {leavers}")
    if attrition_rate is not None:
        context_parts.append(f"attrition rate {_fmt_pct(attrition_rate)}")

    if context_parts:
        lines.append("Context: " + ", ".join(context_parts) + ".")

    if drivers:
        if len(drivers) == 1:
            lines.append(f"Primary driver appears to be {drivers[0]}.")
        else:
            lines.append("Main drivers appear to be " + ", ".join(drivers[:-1]) + f", and {drivers[-1]}.")

    if notes:
        lines.append(f"Additional note: {notes}")

    lines.append("This summary is advisory only. Any HR action requires human review and confirmation.")

    return "\n".join(lines)


def summarize_llm(alert: Dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return summarize_offline(alert)

    prompt = {
        "instruction": (
            "Write a brief workforce-risk explanation in plain language. "
            "Do not invent facts. Use only the supplied fields. "
            "Do not include personally identifying information. "
            "End with a short statement that the output is advisory and requires human review."
        ),
        "alert": alert,
    }

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "You summarize structured workforce risk alerts for internal review.",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False),
                },
            ],
            temperature=0.1,
            max_output_tokens=220,
        )

        text = getattr(response, "output_text", "") or ""
        text = text.strip()

        if not text:
            return summarize_offline(alert)

        return text[:1500]

    except Exception:
        return summarize_offline(alert)


def summarize(alert: Dict[str, Any], use_llm: bool = False) -> str:
    safe_alert = sanitize(alert)

    if use_llm:
        return summarize_llm(safe_alert)

    return summarize_offline(safe_alert)

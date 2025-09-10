from __future__ import annotations
import os, textwrap
from typing import Dict, Iterable

MAX_FIELD_LEN = 500
ALLOWLIST = {
    "dept", "weekly_ts", "headcount", "leavers", "attrition_rate",
    "top_drivers", "risk_score", "threshold", "notes"
}

def _clip(s: str, n: int = MAX_FIELD_LEN) -> str:
    s = s.replace("\r", " ").replace("\n", " ").strip()
    return s[:n]

def sanitize(alert: Dict, allowlist: Iterable[str] = ALLOWLIST) -> Dict:
    safe = {}
    for k in allowlist:
        if k in alert:
            v = alert[k]
            if isinstance(v, str):
                safe[k] = _clip(v)
            elif isinstance(v, (int, float)):
                safe[k] = v
            elif isinstance(v, (list, tuple)):
                safe[k] = [_clip(x) if isinstance(x, str) else x for x in v[:20]]
            elif isinstance(v, dict):
                safe[k] = {kk: (_clip(vv) if isinstance(vv, str) else vv) for kk, vv in list(v.items())[:20]}
            else:
                safe[k] = str(v)[:MAX_FIELD_LEN]
    return safe

def summarize_offline(safe_alert: Dict) -> str:
    dept = safe_alert.get("dept", "unknown")
    ts = safe_alert.get("weekly_ts", "n/a")
    risk = safe_alert.get("risk_score", None)
    thr = safe_alert.get("threshold", None)
    lines = [
        f"[PAWN] Dept '{dept}' @ {ts}:",
        f"- Risk score: {risk:.3f}" if isinstance(risk, (int, float)) else "- Risk score: n/a",
        f"- Threshold τ: {thr:.3f}" if isinstance(thr, (int, float)) else "- Threshold τ: n/a",
    ]
    drivers = safe_alert.get("top_drivers", [])
    if isinstance(drivers, (list, tuple)) and drivers:
        lines.append("- Top drivers: " + ", ".join(map(str, drivers[:5])))
    notes = safe_alert.get("notes", None)
    if notes:
        lines.append(f"- Notes: {notes}")
    lines.append("Action: HR review required. Model is advisory; do not take adverse action without human confirmation.")
    return "\n".join(lines)

def summarize(alert: Dict, allowlist: Iterable[str] = ALLOWLIST, use_llm: bool = False) -> str:
    safe_alert = sanitize(alert, allowlist)
    if not use_llm or not os.getenv("OPENAI_API_KEY"):
        return summarize_offline(safe_alert)
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        prompt = (
            "Summarize this workforce risk alert in 3 concise bullet points, "
            "avoiding PII and operational details. End with a single human action.\n\n"
            f"{safe_alert}"
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a terse risk summarizer."},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        return textwrap.shorten(text, width=800, placeholder="…")
    except Exception:
        return summarize_offline(safe_alert)

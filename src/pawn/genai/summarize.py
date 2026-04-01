from .guard import summarize as _summarize


def generate_summary(dept, week, drivers):
    alert = {
        "dept": dept,
        "weekly_ts": week,
        "top_drivers": drivers,
        "warning": True,
    }
    text = _summarize(alert, use_llm=False)
    return [line for line in text.splitlines() if line.strip()]

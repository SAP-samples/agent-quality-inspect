"""
Shared CSS styles for generated HTML pages.
Matches the design system from the main leaderboard dashboard.
"""

SHARED_CSS = """
:root {
  --font:      "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "SF Mono", ui-monospace, "Cascadia Code", monospace;
  --bg:         #fafafa;
  --surface:    #ffffff;
  --border:     #e4e4e7;
  --border-sub: #f4f4f5;
  --text-1: #18181b;
  --text-2: #52525b;
  --text-3: #a1a1aa;
  --brand:    #4f46e5;
  --brand-bg: #eef2ff;
  --score-hi:  #15803d;
  --score-mid: #b45309;
  --score-lo:  #b91c1c;
  --radius:    6px;
  --radius-sm: 4px;
  --shadow:    0 1px 2px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }
body {
  font-family: var(--font);
  font-size: 14px; line-height: 1.5;
  background: var(--bg); color: var(--text-1); min-height: 100vh;
}
.container { max-width: 1280px; margin: 0 auto; padding: 0 32px; }

/* Nav */
.nav {
  height: 48px; background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center;
  position: sticky; top: 0; z-index: 100;
}
.nav-inner { display: flex; align-items: center; justify-content: space-between; width: 100%; }
.nav-brand {
  font-size: 13px; font-weight: 600; color: var(--text-1);
  text-decoration: none; letter-spacing: -0.01em;
}
.nav-brand .nav-divider { color: var(--border); margin: 0 8px; font-weight: 400; }
.nav-brand .nav-section { color: var(--text-3); font-weight: 400; }

/* Breadcrumb */
.breadcrumb {
  padding: 14px 0 6px; font-size: 12px; color: var(--text-3);
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}
.breadcrumb a { color: var(--text-2); text-decoration: none; }
.breadcrumb a:hover { color: var(--text-1); }
.breadcrumb .sep { color: var(--border); }

/* Page header */
.page-header { padding: 16px 0 28px; }
.page-eyebrow {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--text-2); margin-bottom: 6px;
}
.page-header h1 {
  font-size: 20px; font-weight: 600; letter-spacing: -0.015em;
  color: var(--text-1); margin-bottom: 4px;
}
.page-header .subtitle { font-size: 13px; color: var(--text-2); margin-bottom: 16px; }
.meta-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
.meta-pill {
  padding: 3px 8px; border-radius: var(--radius-sm);
  border: 1px solid var(--border); background: var(--surface);
  font-size: 12px; color: var(--text-2); font-weight: 500;
}
.metrics-row { display: flex; gap: 12px; flex-wrap: wrap; }
.metric-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 20px; min-width: 136px;
  box-shadow: var(--shadow);
}
.metric-card .mc-label {
  font-size: 11px; letter-spacing: 0.03em;
  color: var(--text-2); font-weight: 700; margin-bottom: 6px;
}
.metric-card .mc-value {
  font-size: 22px; font-weight: 600; color: var(--text-1);
  font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
}

/* Section header */
.section-header {
  padding: 24px 0 12px; border-bottom: 1px solid var(--border);
  margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between;
}
.section-header h2 { font-size: 13px; font-weight: 600; color: var(--text-1); }
.section-header .section-sub { font-size: 12px; color: var(--text-2); }

/* Table */
.table-wrap {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); box-shadow: var(--shadow);
  overflow: hidden; overflow-x: auto; margin-bottom: 32px;
}
table { width: 100%; border-collapse: collapse; font-size: 13px; }
thead { background: var(--bg); }
thead tr { border-bottom: 1px solid var(--border); }
thead th {
  padding: 9px 16px; text-align: left; font-size: 11px; font-weight: 600;
  letter-spacing: 0.03em; color: var(--text-2);
  white-space: nowrap;
}
tbody tr { border-bottom: 1px solid var(--border-sub); }
tbody tr:last-child { border-bottom: none; }
tbody tr:hover { background: var(--bg); }
tbody td { padding: 11px 16px; vertical-align: middle; }

.num { font-variant-numeric: tabular-nums; }
.complete   { color: var(--score-hi); font-weight: 600; }
.incomplete { color: var(--score-lo); font-weight: 600; }
.neutral    { color: var(--text-3); }

/* Link button */
.link-btn {
  display: inline-flex; align-items: center;
  padding: 4px 10px; border: 1px solid var(--border);
  border-radius: var(--radius-sm); background: var(--surface);
  font-family: var(--font); font-size: 12px; font-weight: 500;
  color: var(--text-2); text-decoration: none; white-space: nowrap;
  transition: background 0.1s, color 0.1s, border-color 0.1s;
}
.link-btn:hover { background: var(--bg); color: var(--text-1); border-color: var(--text-3); }

/* Score values */
.score-val  { font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums; }
.score-hi   { color: var(--text-1); }
.score-mid  { color: var(--text-1); }
.score-lo   { color: var(--text-1); }
.score-na   { color: var(--text-3); font-weight: 400; }
"""


def esc(s: str) -> str:
    """Escape HTML special characters."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt(v, dp=3) -> str:
    """Format a numeric value."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{dp}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_model(s: str) -> str:
    """Strip provider prefix and capitalise first letter. 'azure/gpt-5' → 'Gpt-5'."""
    if not s:
        return "—"
    name = s.split("/", 1)[-1] if "/" in s else s
    return name[0].upper() + name[1:] if name else "—"


def badge(v) -> str:
    """Generate HTML badge for a score value."""
    if v is None:
        return '<span class="score-na">—</span>'
    n = float(v)
    cls = "score-hi" if n >= 0.75 else ("score-mid" if n >= 0.50 else "score-lo")
    return f'<span class="score-val {cls}">{n:.3f}</span>'


def slugify(s: str) -> str:
    """Convert string to URL-safe slug."""
    import re

    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    return s

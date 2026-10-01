import numpy as np, pandas as pd, plotly.graph_objects as go
from datetime import datetime, timedelta
import bm_routine
from bm_config import get_conn, SEED

PCOL = {"High": "#e11d48", "Medium": "#f59e0b", "Low": "#10b981", "Discard": "#6b7280"}
SKIN_COL = {"acne": "#e11d48", "pigmentation": "#8b5cf6", "redness": "#f97316", "hydration": "#0ea5e9", "overall": "#10b981"}
_L = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), height=340, font=dict(size=13))

def _real_scan(uid):                                   # routines use real scans only, never the synthetic demo history
    with get_conn() as c:
        d = pd.read_sql("SELECT * FROM skin_scans WHERE user_id=? AND COALESCE(image_path,'')!='SYNTHETIC' ORDER BY scan_id DESC LIMIT 1", c, params=(int(uid),))
    return d.iloc[0] if len(d) else None
bm_routine.latest_scan = _real_scan

def get_scans(uid):
    with get_conn() as c: d = pd.read_sql("SELECT * FROM skin_scans WHERE user_id=? ORDER BY scan_date", c, params=(int(uid),))
    d["scan_date"] = pd.to_datetime(d.scan_date); return d

def seed_demo_scans(uid, n=8):                         # SYNTHETIC weekly scans with an illustrative improving trend
    with get_conn() as c: st = c.execute("SELECT skin_type FROM users WHERE user_id=?", (int(uid),)).fetchone()[0]
    rng = np.random.default_rng(SEED + int(uid)); t0 = datetime.now() - timedelta(weeks=n - 1)
    b = {k: rng.uniform(lo, hi) for k, (lo, hi) in dict(acne=(35, 60), pigmentation=(30, 55), redness=(25, 50), hydration=(40, 60)).items()}
    dr = dict(acne=-2.5, pigmentation=-1.2, redness=-1.5, hydration=2.0); rows = []
    for i in range(n):
        v = {k: float(np.clip(b[k] + dr[k] * i + rng.normal(0, 3), 0, 100)) for k in b}
        ov = float(np.mean([100 - v["acne"], 100 - v["pigmentation"], 100 - v["redness"], v["hydration"]]))
        rows.append((int(uid), (t0 + timedelta(weeks=i)).strftime("%Y-%m-%d %H:%M:%S"), st, round(v["acne"], 1), round(v["pigmentation"], 1),
                     round(v["redness"], 1), round(v["hydration"], 1), round(ov, 1), "SYNTHETIC"))
    with get_conn() as c:
        c.execute("DELETE FROM skin_scans WHERE user_id=? AND image_path='SYNTHETIC'", (int(uid),))
        c.executemany("INSERT INTO skin_scans(user_id,scan_date,skin_type,acne,pigmentation,redness,hydration,overall,image_path) VALUES(?,?,?,?,?,?,?,?,?)", rows)

def _empty(msg):
    f = go.Figure(); f.add_annotation(text=msg, showarrow=False, font=dict(size=15, color="#6b7280"))
    f.update_layout(**_L); f.update_xaxes(visible=False); f.update_yaxes(visible=False); return f

def fig_status(rk):
    c = rk.priority.value_counts().reindex(["High", "Medium", "Low", "Discard"]).dropna()
    f = go.Figure(go.Pie(labels=c.index, values=c.values, hole=.6, marker=dict(colors=[PCOL[k] for k in c.index]), textinfo="label+value"))
    return f.update_layout(title="Inventory status by priority", showlegend=False, **_L)

def fig_days(rk):
    d = rk.sort_values("days_left", ascending=False)   # most urgent ends up at the top
    f = go.Figure(go.Bar(y=d["name"], x=d.days_left, orientation="h", marker_color=[PCOL[p] for p in d.priority],
                         text=d.days_left.astype(int).astype(str) + " d", textposition="outside"))
    f.add_vline(x=30, line_dash="dash", line_color="#9ca3af", annotation_text="30-day alert")
    return f.update_layout(title="Days of shelf life left", yaxis=dict(automargin=True), **{**_L, "height": max(320, 34 * len(d) + 110)})

def fig_waste(rk):
    d = rk[rk.days_left > 0].groupby("category").waste_pct.mean().sort_values()
    if d.empty: return _empty("No active products")
    f = go.Figure(go.Bar(x=d.values, y=d.index, orientation="h", marker_color="#f59e0b", text=d.round(0).astype(int).astype(str) + "%", textposition="outside"))
    return f.update_layout(title="Waste risk by category (% expected to expire unused)", xaxis=dict(range=[0, 110]), **_L)

def fig_alerts(al):
    if al.empty: return _empty("No alerts - inventory looks healthy")
    SC = {"Critical": "#e11d48", "High": "#f97316", "Medium": "#f59e0b", "Low": "#10b981"}
    d = al.groupby(["alert_type", "severity"]).size().reset_index(name="n")
    f = go.Figure([go.Bar(name=s, x=g.alert_type, y=g.n, marker_color=SC[s]) for s, g in d.groupby("severity")])
    return f.update_layout(title="Alerts by type", barmode="stack", **_L)

def fig_trend(d):
    if len(d) < 2: return _empty("Need 2+ skin scans: scan in the Skin tab or add demo history")
    f = go.Figure([go.Scatter(x=d.scan_date, y=d[k], name=k.capitalize(), mode="lines+markers", line=dict(color=c, width=4 if k == "overall" else 2)) for k, c in SKIN_COL.items()])
    return f.update_layout(title="Skin score trend", yaxis=dict(range=[0, 100], title="score"), legend=dict(orientation="h", y=-.2), **_L)

def trend_md(sc):
    if len(sc) < 2: return "_Run at least two scans (or add demo history) to see trends._"
    a, b = sc.iloc[0], sc.iloc[-1]
    def d(k, hb):
        x = b[k] - a[k]; return f"{k.capitalize()} {x:+.0f} " + ("✅" if abs(x) >= 2 and (x > 0) == hb else "⚠️" if abs(x) >= 2 else "➖")
    s = f"**Change since first scan** ({a.scan_date.date()} to {b.scan_date.date()}): " + " · ".join([d("acne", False), d("pigmentation", False), d("redness", False), d("hydration", True), d("overall", True)])
    s += "\n\nAcne, pigmentation and redness: lower is better. Hydration and overall: higher is better."
    return s + ("\n\n_Includes synthetic demo scans - illustrative only._" if (sc.image_path == "SYNTHETIC").any() else "")

def kpi_html(rk, al, uid):
    live = rk[rk.days_left > 0]; sc = get_scans(uid)
    it = [("Products", len(rk), "#6366f1"), ("Expiring in 30 days", int(((rk.days_left > 0) & (rk.days_left <= 30)).sum()), "#f59e0b"),
          ("Expired", int((rk.days_left <= 0).sum()), "#e11d48"), ("Alerts", len(al), "#f97316"),
          ("Avg waste risk", f"{live.waste_pct.mean():.0f}%" if len(live) else "-", "#8b5cf6"),
          ("Latest skin score", f"{sc.overall.iloc[-1]:.0f}/100" if len(sc) else "-", "#10b981")]
    return '<div style="display:flex;gap:12px;flex-wrap:wrap">' + "".join(
        f'<div style="flex:1;min-width:130px;padding:14px 16px;border-radius:14px;background:{c}14;border:1px solid {c}33">'
        f'<div style="font-size:12px;color:#6b7280">{t}</div><div style="font-size:26px;font-weight:700;color:{c}">{v}</div></div>' for t, v, c in it) + "</div>"

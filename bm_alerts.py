import itertools, numpy as np, pandas as pd
from bm_config import get_conn
from bm_ml import predict_shelf_life, explain, risk_level, load_inventory

W = dict(urgency=.5, waste=.3, stock=.2)     # priority score weights (tunable, sum = 1)
HIGH, MED = .55, .30                         # score thresholds
WASTE_MIN, UNUSED_USES = 25, 1               # % of container wasted; uses/week below which a product is "unused"
BASE_ING = {"water", "glycerin", "fragrance", "phenoxyethanol"}
TIERS = [(0, "Expired", "Critical", "has expired"), (7, "Expiry-7", "High", "expires within 7 days"),
         (15, "Expiry-15", "Medium", "expires within 15 days"), (30, "Expiry-30", "Low", "expires within 30 days")]
SEV = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

def enrich(inv, ref=None):
    ref = pd.Timestamp(ref) if ref is not None else pd.Timestamp.today().normalize()
    d = inv.copy()
    d["days_open"] = (ref - pd.to_datetime(d.open_date)).dt.days.clip(lower=0)
    d["label_days"] = (pd.to_datetime(d.expiry_date) - ref).dt.days
    d = predict_shelf_life(d, ref)
    d["days_left"] = np.minimum(d.pred_days, d.label_days).clip(lower=0)   # earlier of ML and label
    d["risk"] = d.days_left.map(risk_level)
    rate = ((100 - d.qty_left_pct) / d.days_open.clip(lower=1)).where(d.days_open >= 14)  # % of container per day
    need = d.qty_left_pct / rate.clip(lower=1e-3)                                         # days needed to finish it
    d["waste_pct"] = (d.qty_left_pct * (1 - d.days_left / need)).clip(lower=0).fillna(0).round(0)
    return d

def _actives(s): return {i.strip().lower() for i in str(s).split(",")} - BASE_ING

def build_alerts(d):
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"); out = []
    add = lambda r, t, s, m: out.append(dict(user_id=int(r.user_id), product_id=int(r.product_id),
                                             alert_type=t, severity=s, message=m, created_at=now))
    for r in d.itertuples():
        for lim, t, s, txt in TIERS:
            if r.days_left <= lim:
                add(r, t, s, f"{r.name} {txt}. Model estimate {r.pred_days}d, label {r.label_days}d, confidence {r.confidence:.0f}%.")
                break
        if r.days_left <= 0: continue
        if r.usage_per_week < UNUSED_USES and r.days_open >= 30:
            add(r, "Unused", "Medium", f"{r.name} is barely used ({r.usage_per_week:g}/week over {r.days_open} days) with {r.qty_left_pct:.0f}% left and ~{r.days_left}d of life. Use it or let it go.")
        elif r.waste_pct >= WASTE_MIN:
            add(r, "Waste risk", "High" if r.waste_pct >= 60 else "Medium", f"At your current pace ~{r.waste_pct:.0f}% of {r.name} will expire unused (~{r.days_left}d left).")
    live = d[d.days_left > 0]
    for (uid, cat), g in live.groupby(["user_id", "category"]):
        for a, b in itertools.combinations(list(g.itertuples()), 2):
            sh = _actives(a.ingredients) & _actives(b.ingredients)
            if sh:
                f = min((a, b), key=lambda r: r.days_left)
                add(f, "Duplicate", "Low", f"Duplicate {cat.lower()}s: {a.name} (#{a.product_id}) and {b.name} (#{b.product_id}) share {', '.join(sorted(sh))}. Use {f.name} (#{f.product_id}) first ({f.days_left}d left) and avoid buying more.")
    try: from bm_rules import find_conflicts          # provided in Stage 3
    except ImportError: find_conflicts = None
    if find_conflicts:
        for uid, g in live.groupby("user_id"):
            for c in find_conflicts(g):
                out.append(dict(user_id=int(uid), product_id=int(c["product_id"]), alert_type="Routine conflict",
                                severity=c["severity"], message=c["message"], created_at=now))
    a = pd.DataFrame(out, columns=["user_id", "product_id", "alert_type", "severity", "message", "created_at"])
    return a.assign(_s=a.severity.map(SEV)).sort_values(["user_id", "_s"]).drop(columns="_s").reset_index(drop=True)

def _reason(r):
    if r.priority == "Discard": return "Expired by label or model estimate - stop using and replace."
    if r.priority == "Low": return f"No rush: ~{r.days_left} days of shelf life left, {r.qty_left_pct:.0f}% remaining."
    c = {"u": W["urgency"]*r.urgency, "w": W["waste"]*r.waste_pct/100, "s": W["stock"]*r.qty_left_pct/100}
    txt = {"u": f"expires in ~{r.days_left} days", "w": f"~{r.waste_pct:.0f}% would go unused at your current pace",
           "s": f"{r.qty_left_pct:.0f}% still left"}
    top = [txt[k] for k in sorted(c, key=c.get, reverse=True) if c[k] >= .04][:3]
    return ("Use first: " if r.priority == "High" else "Use soon: ") + "; ".join(top) + "."

def rank_products(d, explain_high=False):
    d = d.copy(); d["urgency"] = (1 - d.days_left / 90).clip(0, 1)
    d["score"] = (W["urgency"]*d.urgency + W["waste"]*d.waste_pct/100 + W["stock"]*d.qty_left_pct/100).round(3)
    d["priority"] = np.select([d.days_left <= 0, d.score >= HIGH, d.score >= MED], ["Discard", "High", "Medium"], "Low")
    d["reason"] = [_reason(r) for r in d.itertuples()]
    if explain_high:                                   # SHAP drivers of the shelf-life estimate, High items only
        m = d.priority == "High"; d["why_ml"] = ""
        if m.any(): d.loc[m, "why_ml"] = explain(d[m])
    o = d.priority.map({"High": 0, "Medium": 1, "Low": 2, "Discard": 3})
    return d.assign(_o=o).sort_values(["_o", "score"], ascending=[True, False]).drop(columns="_o")

def save_alerts(a, user_ids=None):
    with get_conn() as c:
        if user_ids is None: c.execute("DELETE FROM alerts")
        else: c.execute(f"DELETE FROM alerts WHERE user_id IN ({','.join(str(int(u)) for u in user_ids)})")
        a.to_sql("alerts", c, if_exists="append", index=False)

def run_user(user_id, ref=None, explain_high=True, save=True):
    d = enrich(load_inventory(user_id), ref); a = build_alerts(d)
    if save: save_alerts(a, [user_id])
    return rank_products(d, explain_high), a

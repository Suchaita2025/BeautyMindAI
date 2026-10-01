import numpy as np, pandas as pd, joblib
from bm_config import P, get_conn

CATS = ["Cleanser","Toner","Serum","Moisturizer","Sunscreen","Eye Cream","Face Mask","Exfoliant","Spot Treatment"]
STORAGES = ["Drawer","Shelf","Bathroom","Fridge","Sunlit"]
CLIMATES = ["Humid","Hot-Dry","Temperate","Cold"]
JARS = ["Moisturizer","Eye Cream","Face Mask"]
PRESERVATIVES = ["phenoxyethanol","sodium benzoate","potassium sorbate","ethylhexylglycerin"]
FLAGS = {"vitc":["ascorbic"], "retinol":["retinol","retinal"], "aha":["glycolic","lactic","mandelic"],
         "bha":["salicylic"], "bpo":["benzoyl peroxide"]}
_k = lambda s: str(s).replace(" ", "_")

def featurize(df, ref=None):
    d = df.copy()
    if "days_open" not in d:
        ref = pd.Timestamp(ref) if ref is not None else pd.Timestamp.today().normalize()
        d["days_open"] = (ref - pd.to_datetime(d["open_date"])).dt.days.clip(lower=0)
    ing = d["ingredients"].fillna("").str.lower()
    X = pd.DataFrame({"pao_months": d["pao_months"], "days_open": d["days_open"],
        "label_days_left": d["pao_months"]*30 - d["days_open"],
        "qty_left_pct": d["qty_left_pct"], "usage_per_week": d["usage_per_week"],
        "jar_type": d["category"].isin(JARS).astype(int),
        "preservative_free": (~ing.apply(lambda s: any(p in s for p in PRESERVATIVES))).astype(int)})
    for f, keys in FLAGS.items(): X["has_"+f] = ing.apply(lambda s: int(any(k in s for k in keys)))
    for c in CATS: X["cat_"+_k(c)] = (d["category"] == c).astype(int)
    for s in STORAGES: X["storage_"+_k(s)] = (d["storage"] == s).astype(int)
    for c in CLIMATES: X["climate_"+_k(c)] = (d["climate"] == c).astype(int)
    return X

def risk_level(days): return "Expired" if days <= 0 else "High" if days <= 14 else "Medium" if days <= 45 else "Low"

def load_inventory(user_id=None):
    q = "SELECT p.*, u.climate FROM products p JOIN users u USING(user_id)"
    if user_id: q += f" WHERE p.user_id={int(user_id)}"
    with get_conn() as c: return pd.read_sql(q, c)

def load_models(): return joblib.load(f"{P['models']}/shelf_models.pkl")

def predict_shelf_life(df, ref=None):
    M = load_models(); X = featurize(df, ref)
    preds = np.column_stack([M[k].predict(X) for k in ("rf", "gb", "xgb")])
    best = M[M["best"]].predict(X)
    out = df.copy()
    out["pred_days"] = np.maximum(best, 0).round().astype(int)
    out["confidence"] = (100*np.exp(-preds.std(axis=1)/15)).round(0)  # heuristic: agreement of the 3 models
    out["risk"] = [risk_level(v) for v in best]
    return out

LABEL = {"days_open":"days since opening","label_days_left":"days left on PAO label","pao_months":"PAO months",
 "qty_left_pct":"amount left %","usage_per_week":"uses/week","jar_type":"jar-type packaging",
 "preservative_free":"preservative-free formula","has_vitc":"Vitamin C","has_retinol":"retinol",
 "has_aha":"AHA","has_bha":"BHA","has_bpo":"benzoyl peroxide"}
_ex = {}
def explain(df, ref=None, k=3):
    import shap
    if "e" not in _ex: M = load_models(); _ex["e"] = shap.TreeExplainer(M[M["best"]])
    X = featurize(df, ref); sv = _ex["e"].shap_values(X); out = []
    for i in range(len(X)):
        s = sv[i].copy(); parts = []
        for j, f in enumerate(X.columns):   # ignore absent binary flags (value 0)
            if f.startswith(("cat_","storage_","climate_","has_")) or f in ("jar_type","preservative_free"):
                if X.iat[i, j] == 0: s[j] = 0
        for j in np.argsort(-np.abs(s))[:k]:
            f = X.columns[j]
            if f.startswith(("cat_","storage_","climate_")):
                kind, nm = f.split("_", 1); lab = f"{kind} = {nm.replace('_',' ')}"
            else: lab = LABEL[f] + ("" if X.iat[i, j] in (0, 1) and f not in ("qty_left_pct",) else f" = {X.iat[i, j]:g}")
            parts.append(f"{lab} {'shortens' if s[j] < 0 else 'extends'} life by ~{abs(s[j]):.0f} days vs. average")
        out.append("; ".join(parts))
    return out

import json, itertools, functools, numpy as np, pandas as pd
from datetime import datetime
import bm_ml
from bm_config import get_conn
from bm_ml import load_inventory
from bm_alerts import enrich, rank_products
from bm_rules import pair_rules, allergy_hits, skin_cautions, allowed_slots, classes_of, LBL
bm_ml.load_models = functools.lru_cache(maxsize=1)(bm_ml.load_models)      # load the shelf-life models once, not per call

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BASE = {"AM": ["Cleanser", "Toner", "Serum", "Eye Cream", "Moisturizer", "Sunscreen"], "PM": ["Cleanser", "Toner", "Serum", "Eye Cream", "Moisturizer"]}
ORDER = ["Cleanser", "Exfoliant", "Face Mask", "Toner", "Serum", "Spot Treatment", "Eye Cream", "Moisturizer", "Sunscreen"]
STRONG, RINSE = {"retinol", "aha", "bha", "bpo"}, {"Cleanser", "Face Mask"}     # rinse-off products skip same-slot clash checks
PAT = {1: [6], 2: [1, 4], 3: [0, 2, 4], 4: [0, 2, 4, 6]}                          # evenly spread nights for x uses/week
PRIO = {"High": 1.5, "Medium": .75, "Low": 0}; SCAN_T = 45
CONCERN_KEYS = {"Acne": ["salicylic", "benzoyl", "sulfur", "niacinamide"], "Pigmentation": ["ascorbic", "niacinamide", "glycolic", "retinol"],
    "Redness": ["niacinamide", "ceramide", "squalane"], "Dryness": ["hyaluronic", "ceramide", "squalane", "shea"],
    "Fine lines": ["retinol", "peptide", "ascorbic"], "Dullness": ["ascorbic", "glycolic", "lactic", "niacinamide"], "Oiliness": ["niacinamide", "salicylic", "kaolin"]}

def latest_scan(uid):
    with get_conn() as c: d = pd.read_sql("SELECT * FROM skin_scans WHERE user_id=? ORDER BY scan_id DESC LIMIT 1", c, params=(int(uid),))
    return d.iloc[0] if len(d) else None

def clash(a, b):                                       # High/Medium rule hit between two products, else None
    if a["category"] in RINSE or b["category"] in RINSE: return None
    h = [x for x in pair_rules(a["ingredients"], b["ingredients"]) if x["severity"] in ("High", "Medium")]
    return h[0] if h else None

def slots_of(r):
    s = allowed_slots(r["category"], r["ingredients"])
    if r["category"] in ("Exfoliant", "Face Mask"): s = (s & {"PM"}) or {"PM"}
    return {"AM"} if "vitc" in r["cls"] and "AM" in s else s

def weekly_freq(r, skin):
    lim = {"retinol": 3, "aha": 2, "bha": 3, "bpo": 3}
    lim.update({"Sensitive": {"retinol": 2, "aha": 1, "bha": 2, "bpo": 2}, "Dry": {"aha": 1, "bha": 2, "bpo": 2}}.get(skin, {}))
    f = [lim[c] for c in r["cls"] if c in lim]
    if r["category"] == "Face Mask": f.append(1 if skin == "Sensitive" else 2)
    return min(f) if f else 2

def fit_adj(skin, cls, ing):
    a = 0.
    if skin == "Sensitive": a -= .8 * len(cls & STRONG)
    if skin == "Dry": a += .4 * (("hyaluronic" in ing) or ("ceramide" in ing)) - .4 * len(cls & {"aha", "bha", "bpo"})
    if skin in ("Oily", "Combination"): a += .3 * (("niacinamide" in ing) or ("salicylic" in ing))
    return a

def _why(o, slot, ds, skin, src):
    p = []
    if o["hitmap"]: p.append("helps with " + "; ".join(f"{c.lower()} ({src[c]}, via {', '.join(k[:2])})" for c, k in o["hitmap"].items()))
    if o["category"] == "Sunscreen": p.append("daily UV protection, applied last in the morning")
    if slot == "PM" and o["cls"] & {"retinol", "aha"}: p.append("night-only: retinol and AHAs raise sun sensitivity")
    if "vitc" in o["cls"]: p.append("morning use, kept away from retinol and benzoyl peroxide")
    if len(ds) < 7 and o["active"]:
        t = f"{len(ds)}x/week ({', '.join(DAYS[d] for d in ds)}) to limit irritation"
        if o.get("_f", len(ds)) < o.get("_f0", len(ds)): t += f"; reduced from {o['_f0']}x to avoid clashing with your other actives"
        p.append(t)
    if o.get("_repl"): p.append(f"replaces {o['_repl']} on those nights")
    if o["priority"] in ("High", "Medium"): p.append(f"use-first: ~{o['days_left']} days of shelf life left")
    p += skin_cautions(skin, o["ingredients"])
    if not p: return "Gentle staple: no allergens or ingredient clashes found."
    s = "; ".join(p); return s[0].upper() + s[1:] + "."

def build_routine(user_id, ref=None, use_scan=True, scan=None, skin_override=None, fill=None):
    with get_conn() as c: u = pd.read_sql("SELECT * FROM users WHERE user_id=?", c, params=(int(user_id),)).iloc[0]
    skin, notes = (skin_override or u["skin_type"]), []
    _inv = load_inventory(user_id)
    rk = rank_products(enrich(_inv, ref)) if len(_inv) else pd.DataFrame({c: pd.Series(dtype=t) for c, t in dict(product_id='int64', name='object', category='object', ingredients='object', days_left='float64', priority='object', qty_left_pct='float64', waste_pct='float64').items()})
    concerns = [x.strip() for x in str(u["concerns"]).split(",") if x.strip() and x.strip() != "None"]; src = {c: "your profile" for c in concerns}
    sc = scan if scan is not None else (latest_scan(user_id) if use_scan else None)
    if sc is not None:
        for c, on in [("Acne", sc["acne"] >= SCAN_T), ("Pigmentation", sc["pigmentation"] >= SCAN_T), ("Redness", sc["redness"] >= SCAN_T), ("Dryness", sc["hydration"] <= 100 - SCAN_T)]:
            if on and c not in src: concerns.append(c); src[c] = "your latest scan"
        if str(sc["skin_type"]).lower() != str(u["skin_type"]).lower(): notes.append(f"Your latest scan suggests {sc['skin_type']} skin; your profile says {u['skin_type']}. This routine follows the scan.")
    if fill is not None: rk = pd.concat([rk, fill(rk, u, skin, concerns)], ignore_index=True)   # suggested products for missing categories
    # ---- eligibility: not expired, allergy-safe, scored
    E, excl = [], []
    for r in rk.to_dict("records"):
        ing = str(r["ingredients"]).lower(); r["cls"] = classes_of(ing); al = allergy_hits(ing, u["allergies"])
        if r["days_left"] <= 0: excl.append((r, "expired", "Expired or past its estimated shelf life - stop using it and replace it.")); continue
        if al: excl.append((r, "allergy", f"Contains {', '.join(al)} from your allergy list - do not use.")); continue
        r["hitmap"] = {c: [k for k in CONCERN_KEYS.get(c, []) if k in ing] for c in concerns}; r["hitmap"] = {c: k for c, k in r["hitmap"].items() if k}
        r["fit"] = len(r["hitmap"]) + fit_adj(skin, r["cls"], ing) + PRIO.get(r["priority"], 0)
        r["active"] = r["category"] in ("Exfoliant", "Face Mask", "Spot Treatment") or (r["category"] != "Cleanser" and bool(r["cls"] & STRONG))
        E.append(r)
    byfit = lambda r: (-r["fit"], r["days_left"], r["product_id"])
    plan = {s: {d: [] for d in range(7)} for s in ("AM", "PM")}; am_serum = []
    # ---- daily base: best conflict-free product per category, per slot
    for slot in ("AM", "PM"):
        chosen = []
        for cat in BASE[slot]:
            cands = sorted([r for r in E if not r["active"] and r["category"] == cat and slot in slots_of(r)], key=byfit)
            if cat == "Serum" and slot == "PM": cands = [r for r in cands if r["product_id"] not in am_serum] or cands
            for r in cands:
                if not any(clash(r, o) for o in chosen): chosen.append(r); break
            if cat == "Serum" and slot == "AM": am_serum = [o["product_id"] for o in chosen if o["category"] == "Serum"]
        for d in range(7): plan[slot][d] = list(chosen)
    # ---- strong actives, masks, exfoliants, spot treatments: scheduled on spread nights without clashes
    taken = set()
    for r in sorted([r for r in E if r["active"]], key=byfit):
        strong = r["cls"] & STRONG
        if r["category"] == "Spot Treatment" and "Acne" not in concerns:
            excl.append((r, "not_needed", "Spot treatment not needed: no acne concern in your profile or latest scan.")); continue
        if strong & taken:
            excl.append((r, "not_needed", f"Not needed: another product already covers {', '.join(LBL[x] for x in sorted(strong & taken))}; doubling up raises irritation.")); continue
        f0, ok = weekly_freq(r, skin), False
        for f in range(f0, 0, -1):
            for slot in sorted(slots_of(r), key=lambda s: s != "PM"):
                cands = []
                for sh in range(7):
                    nights = [(d + sh) % 7 for d in PAT[f]]
                    others = [o for d in nights for o in plan[slot][d] if not (o["category"] == r["category"] and not o["active"])]
                    if any(o["category"] == r["category"] for o in others) or any(clash(r, o) for o in others): continue
                    cands.append((sum(o["active"] for d in nights for o in plan[slot][d]), sh, nights))
                if cands:
                    nights = min(cands)[2]
                    for d in nights:
                        repl = {o["product_id"]: o for o in plan[slot][d] if o["category"] == r["category"] and not o["active"]}
                        if repl: r["_repl"] = next(iter(repl.values()))["name"]
                        plan[slot][d] = [o for o in plan[slot][d] if o["product_id"] not in repl] + [r]
                    taken |= strong; r["_f0"], r["_f"], ok = f0, f, True; break
            if ok: break
        if not ok: excl.append((r, "unschedulable", "Could not be scheduled without a High/Medium ingredient clash with your other products."))
    # ---- items, grid, explanations
    agg = {}
    for slot in ("AM", "PM"):
        for d in range(7):
            for o in plan[slot][d]: agg.setdefault((slot, o["product_id"]), [o, []])[1].append(d)
    items = [dict(slot=s, category=o["category"], product_id=int(pid), name=o["name"], daily=len(ds) == 7, n_days=len(ds),
                  days="Daily" if len(ds) == 7 else ", ".join(DAYS[d] for d in ds), reason=_why(o, s, ds, skin, src)) for (s, pid), (o, ds) in agg.items()]
    items.sort(key=lambda i: (i["slot"], ORDER.index(i["category"]), -i["n_days"], i["product_id"]))
    used = {i["product_id"]: agg[(i["slot"], i["product_id"])][0] for i in items}
    where = {pid: ", ".join(f"{i['slot']} {i['days']}" for i in items if i["product_id"] == pid) for pid in used}
    for a, b in itertools.combinations(used.values(), 2):
        h = clash(a, b)
        if h: x, y = h["pair"]; notes.append(f"Kept apart: {a['name']} ({where[a['product_id']]}) and {b['name']} ({where[b['product_id']]}) - {LBL[x]} + {LBL[y]}: {h['why']}.")
    done = {r["product_id"] for r, _, _ in excl}
    for r in E:
        if r["product_id"] not in used and r["product_id"] not in done:
            alt = [i["name"] for i in items if i["category"] == r["category"]]
            excl.append((r, "backup", f"Backup: {alt[0]} fits your skin and concerns better right now." if alt else "Not scheduled: no suitable slot in this routine."))
    for slot, cat, hint in [("AM", "Cleanser", ""), ("PM", "Cleanser", ""), ("AM", "Moisturizer", ""), ("PM", "Moisturizer", ""), ("AM", "Sunscreen", " (the most important daily step)")]:
        miss = [d for d in range(7) if not any(o["category"] == cat for o in plan[slot][d])]
        if miss: notes.append(f"Gap: no usable {cat.lower()} for {slot} on {'every day' if len(miss) == 7 else ', '.join(DAYS[d] for d in miss)}{hint} - consider adding one.")
    if u["lifestyle"] == "Outdoor/active": notes.append("Tip: reapply sunscreen every 2-3 hours when outdoors.")
    if u["climate"] in ("Hot-Dry", "Cold"): notes.append("Tip: apply moisturizer on slightly damp skin to lock in water in dry or cold air.")
    if u["climate"] == "Humid": notes.append("Tip: humid climate speeds spoilage - keep lids closed and products out of the bathroom.")
    grid = pd.DataFrame({s: [" > ".join(o["name"] for o in sorted(plan[s][d], key=lambda o: ORDER.index(o["category"]))) for d in range(7)] for s in ("AM", "PM")}, index=DAYS)
    summary = (f"Routine for {u['name']} ({skin} skin, {u['climate']} climate). Concerns considered: "
               + (", ".join(f"{c} ({src[c]})" for c in concerns) or "none listed") + f". Uses {len(used)} of {len(rk)} owned products; {len(excl)} not used (reasons below).")
    return dict(user=dict(name=u["name"], skin_type=skin, allergies=u["allergies"], climate=u["climate"], concerns=concerns), items=items, grid=grid, notes=notes, summary=summary,
                excluded=[dict(product_id=int(r["product_id"]), name=r["name"], category=r["category"], kind=k, reason=t) for r, k, t in excl], _plan=plan)

def validate(R, allergies):                            # independent safety check of the generated plan
    v = []
    for slot in ("AM", "PM"):
        for d in range(7):
            ps = R["_plan"][slot][d]
            if len({a["category"] for a in ps}) != len(ps): v.append(f"duplicate category {slot} {DAYS[d]}")
            for i, a in enumerate(ps):
                if a["days_left"] <= 0: v.append(f"expired: {a['name']}")
                if allergy_hits(a["ingredients"], allergies): v.append(f"allergen: {a['name']}")
                if slot == "AM" and a["cls"] & {"retinol", "aha"}: v.append(f"AM retinol/AHA: {a['name']}")
                if slot == "PM" and a["category"] == "Sunscreen": v.append(f"PM sunscreen: {a['name']}")
                v += [f"clash: {a['name']} + {b['name']} ({slot} {DAYS[d]})" for b in ps[i + 1:] if clash(a, b)]
    return v

def render_text(R):
    L = [R["summary"], ""]
    for slot in ("AM", "PM"):
        L.append(f"===== {slot} ROUTINE =====")
        for n, it in enumerate([x for x in R["items"] if x["slot"] == slot], 1): L.append(f"{n}. {it['category']}: {it['name']} [{it['days']}]\n     why: {it['reason']}")
        L.append("")
    L += ["===== WEEKLY GRID =====", R["grid"].to_string(), "", "===== NOT USED ====="]
    L += [f"- {e['name']} ({e['kind']}): {e['reason']}" for e in R["excluded"]] or ["- none"]
    L += ["", "===== NOTES ====="] + [f"- {n}" for n in R["notes"]]
    return "\n".join(L)

def save_routine(user_id, R):
    with get_conn() as c:
        c.execute("INSERT INTO routines(user_id, created_at, routine_json, explanation) VALUES(?,?,?,?)", (int(user_id), datetime.now().strftime("%Y-%m-%d %H:%M"),
                  json.dumps({k: R[k] for k in ("user", "items", "excluded", "notes")}, default=str), render_text(R)))

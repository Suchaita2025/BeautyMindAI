import os
"""Starter-routine suggestions for BeautyMind AI.
Generic, ingredient-based product TYPES (not brands, not endorsements) used to fill gaps when a user has no suitable product.
Suggestions are filtered by allergies, age and concerns, then planned by the same rule engine as owned products."""
import pandas as pd
from bm_rules import allergy_hits

# (category, name, ingredients, only_if_any_of_these_concerns (None = always eligible), minimum age)
CATALOG = [
 ("Cleanser", "Gentle Hydrating Cleanser", "Water, Glycerin, Ceramides, Hyaluronic Acid", None, 0),
 ("Cleanser", "Clarifying Salicylic Acid Cleanser", "Water, Glycerin, Salicylic Acid", {"Acne", "Oiliness"}, 0),
 ("Cleanser", "Balancing Gel Cleanser", "Water, Glycerin, Niacinamide", None, 0),
 ("Serum", "Hyaluronic Acid Hydration Serum", "Water, Glycerin, Hyaluronic Acid", None, 0),
 ("Serum", "Niacinamide Balancing Serum", "Water, Glycerin, Niacinamide", None, 0),
 ("Serum", "Vitamin C Brightening Serum", "Water, Glycerin, Ascorbic Acid", {"Pigmentation", "Dullness", "Fine lines"}, 0),
 ("Serum", "Retinol Night Serum", "Water, Squalane, Retinol", {"Fine lines", "Pigmentation"}, 25),
 ("Moisturizer", "Ceramide Barrier Cream", "Water, Glycerin, Ceramides, Squalane, Shea Butter", None, 0),
 ("Moisturizer", "Lightweight Gel Moisturizer", "Water, Glycerin, Hyaluronic Acid, Niacinamide", None, 0),
 ("Sunscreen", "Mineral SPF 50 Sunscreen", "Zinc Oxide, Glycerin, Squalane", None, 0),
 ("Sunscreen", "Lightweight Gel SPF 50 Sunscreen", "Avobenzone, Glycerin, Niacinamide", None, 0),
 ("Spot Treatment", "Benzoyl Peroxide Spot Gel", "Water, Glycerin, Benzoyl Peroxide", {"Acne"}, 0),
 ("Spot Treatment", "Sulfur Spot Treatment", "Kaolin Clay, Glycerin, Sulfur", {"Acne"}, 0),
]
LOOK = {-1000 + i: ", ".join(x.strip() for x in c[2].split(",") if x.strip() not in ("Water", "Glycerin")) for i, c in enumerate(CATALOG)}

def make_fill():
    """Returns fill(rk, user_row, skin, concerns) -> DataFrame of suggested products for the categories the user is missing."""
    def fill(rk, u, skin, concerns):
        usable = rk[rk.days_left > 0] if len(rk) else rk
        have = {c for c, i in zip(usable.category, usable.ingredients) if not allergy_hits(i, u["allergies"])}
        need = {"Cleanser", "Moisturizer", "Sunscreen"} - have
        if "Serum" not in have: need.add("Serum")
        if "Acne" in concerns and "Spot Treatment" not in have: need.add("Spot Treatment")
        cs, age, rows, spot = set(concerns), int(u["age"] or 0), [], False
        for i, (cat, name, ing, only, min_age) in enumerate(CATALOG):
            if cat not in need or (only and not only & cs) or age < min_age or allergy_hits(ing, u["allergies"]): continue
            if name == "Retinol Night Serum" and "Acne" in cs: continue        # acne-first: no retinoid next to a benzoyl peroxide plan
            if (skin == "Sensitive" or u["skin_type"] == "Sensitive") and name in ("Retinol Night Serum", "Benzoyl Peroxide Spot Gel"): continue   # gentlest options if EITHER profile or scan says sensitive
            if cat == "Spot Treatment":                                         # at most one spot treatment
                if spot: continue
                spot = True
            rows.append(dict(product_id=-1000 + i, user_id=int(u["user_id"]), name=name, brand="Suggested", category=cat, ingredients=ing,
                             days_left=365, priority="Low", qty_left_pct=100.0, waste_pct=0.0))
        return pd.DataFrame(rows)
    return fill

def apply_routine_patch(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm_routine.py")):
    """One-time, idempotent edit of bm_routine.py: adds scan / skin_override / fill parameters and handles users with no products."""
    s = open(path).read()
    if "fill=None" in s: return "already patched"
    reps = [
     ("def build_routine(user_id, ref=None, use_scan=True):", "def build_routine(user_id, ref=None, use_scan=True, scan=None, skin_override=None, fill=None):"),
     ('skin, notes = u["skin_type"], []', 'skin, notes = (skin_override or u["skin_type"]), []'),
     ("    rk = rank_products(enrich(load_inventory(user_id), ref))\n",
      "    _inv = load_inventory(user_id)\n"
      "    rk = rank_products(enrich(_inv, ref)) if len(_inv) else pd.DataFrame({c: pd.Series(dtype=t) for c, t in dict(product_id='int64', name='object', category='object', ingredients='object', days_left='float64', priority='object', qty_left_pct='float64', waste_pct='float64').items()})\n"),
     ("sc = latest_scan(user_id) if use_scan else None", "sc = scan if scan is not None else (latest_scan(user_id) if use_scan else None)"),
     ("    # ---- eligibility: not expired, allergy-safe, scored\n",
      "    if fill is not None: rk = pd.concat([rk, fill(rk, u, skin, concerns)], ignore_index=True)   # suggested products for missing categories\n"
      "    # ---- eligibility: not expired, allergy-safe, scored\n")]
    for old, new in reps:
        assert old in s, f"patch anchor not found: {old[:60]!r}. Your bm_routine.py differs from the Stage 5 version; paste its build_routine header here."
        s = s.replace(old, new, 1)
    open(path, "w").write(s); return "patched"

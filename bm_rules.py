import itertools

KEYS = {"vitc": ["ascorbic"], "retinol": ["retinol", "retinal"], "aha": ["glycolic", "lactic", "mandelic"],
        "bha": ["salicylic"], "bpo": ["benzoyl peroxide"], "peptides": ["peptide"], "sulfur": ["sulfur"]}
LBL = {"vitc": "Vitamin C", "retinol": "Retinol", "aha": "AHA", "bha": "BHA", "bpo": "Benzoyl peroxide", "peptides": "Peptides"}
SEV = {"High": 0, "Medium": 1, "Low": 2}
FIX = {"split": "use one in the AM and the other in the PM", "alternate": "use them on alternate nights"}

RULES = [  # (class A, class B, severity, fix, reason)
 ("retinol", "aha", "High", "alternate", "both exfoliate and irritate; stacking raises peeling and barrier damage"),
 ("retinol", "bpo", "High", "split", "benzoyl peroxide can degrade some retinoids and the pair is very drying"),
 ("vitc", "bpo", "High", "split", "benzoyl peroxide oxidises vitamin C and cancels its benefit"),
 ("retinol", "bha", "Medium", "alternate", "combined exfoliation and dryness raise irritation risk"),
 ("vitc", "retinol", "Medium", "split", "vitamin C suits mornings and retinol nights; layering raises irritation"),
 ("vitc", "aha", "Medium", "split", "stacking low-pH acids raises irritation"),
 ("aha", "bha", "Medium", "alternate", "double chemical exfoliation risks over-exfoliation"),
 ("aha", "bpo", "Medium", "alternate", "both are drying and irritating when layered"),
 ("bha", "bpo", "Low", "alternate", "combined dryness; watch for tightness"),
 ("peptides", "aha", "Low", "split", "strong acids may reduce peptide stability"),
 ("peptides", "vitc", "Low", "split", "low-pH vitamin C may reduce peptide stability")]

ALLERGY_KEYS = {"fragrance": ["fragrance", "parfum"], "benzoyl peroxide": ["benzoyl peroxide"],
                "salicylic acid": ["salicylic"], "retinol": ["retinol", "retinal"], "sulfur": ["sulfur"]}

def classes_of(ing):
    s = str(ing).lower(); return {c for c, ks in KEYS.items() if any(k in s for k in ks)}

def pair_rules(ing_a, ing_b):
    ca, cb = classes_of(ing_a), classes_of(ing_b)
    hits = [dict(pair=(x, y), severity=s, fix=f, why=w) for x, y, s, f, w in RULES
            if (x in ca and y in cb) or (y in ca and x in cb)]
    return sorted(hits, key=lambda h: SEV[h["severity"]])

def conflict_pairs(df):
    out = []
    for a, b in itertools.combinations(list(df.itertuples()), 2):
        h = pair_rules(a.ingredients, b.ingredients)
        if not h: continue
        anchor = min((a, b), key=lambda r: getattr(r, "days_left", 0))
        top = h[0]; x, y = top["pair"]
        msg = (f"{a.name} + {b.name}: {LBL[x]} + {LBL[y]} - {top['why']}. Fix: {FIX[top['fix']]}."
               + (f" Also: {h[1]['why']}." if len(h) > 1 else ""))
        out.append(dict(product_id=int(anchor.product_id), a_id=int(a.product_id), b_id=int(b.product_id),
                        severity=top["severity"], fix=top["fix"], message=msg))
    return out

find_conflicts = conflict_pairs    # hook used by bm_alerts.build_alerts

def allergy_hits(ing, allergies):
    ing = str(ing).lower(); hits = []
    for a in str(allergies).lower().split(","):
        a = a.strip()
        if a and a != "none" and any(k in ing for k in ALLERGY_KEYS.get(a, [a])): hits.append(a)
    return hits

def skin_cautions(skin_type, ing):
    c = classes_of(ing); out = []
    if skin_type == "Sensitive" and c & {"retinol", "aha", "bha", "bpo"}:
        out.append("strong active on sensitive skin: patch-test and start 1-2 times per week")
    if skin_type == "Dry" and c & {"aha", "bha", "bpo"}:
        out.append("can be drying for dry skin: follow with a moisturizer")
    return out

def allowed_slots(category, ing):
    s = {"AM", "PM"}; c = classes_of(ing)
    if category == "Sunscreen": s = {"AM"}
    if c & {"retinol", "aha"}: s &= {"PM"}       # photosensitising or light-unstable actives: night only
    return s or {"PM"}

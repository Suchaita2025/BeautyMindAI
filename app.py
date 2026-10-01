
import os
import html
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from bm_config import P, get_conn
from bm_ml import CATS, STORAGES, CLIMATES
from bm_alerts import run_user
import bm_routine as BR
import bm_dash as D
from bm_skin import analyze, save_scan
from bm_recommend import make_fill, LOOK

st.set_page_config(
    page_title="BeautyMind AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Styling --------------------
st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 1.2rem;}
.hero {
    padding: 30px 34px; border-radius: 24px; color: white;
    background: linear-gradient(120deg,#e11d48,#a21caf 55%,#6d28d9);
    margin-bottom: 18px;
}
.hero h1 {margin:0; font-size:38px; font-weight:800;}
.hero p {margin:7px 0 0; opacity:.94; font-size:16px;}
.smallnote {color:#6b7280; font-size:13px;}
.metric-card {
    padding:16px; border-radius:16px; background:#fafafa;
    border:1px solid #e5e7eb; margin-bottom:10px;
}
.section-card {
    padding:18px; border-radius:18px; background:white;
    border:1px solid #e5e7eb; margin-bottom:14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>✨ BeautyMind AI</h1>
<p>Explainable decision support for cosmetic expiry management and personalised skincare routines</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Constants --------------------
SKIN = ["Normal", "Dry", "Oily", "Combination", "Sensitive"]
LIFESTYLES = ["Outdoor/active", "Frequent traveler", "Work-from-home", "Student", "Other"]
CONCERNS = ["Acne", "Pigmentation", "Fine lines", "Oiliness", "Dullness", "Dryness", "Redness"]
ALLERGENS = ["Fragrance", "Salicylic Acid", "Retinol", "Benzoyl Peroxide", "AHA", "Vitamin C"]

# -------------------- Helpers --------------------
def q(sql, params=()):
    with get_conn() as c:
        return pd.read_sql(sql, c, params=params)

def user_row(uid):
    if uid is None:
        raise ValueError("Enter a user ID.")
    d = q("SELECT * FROM users WHERE user_id=?", (int(uid),))
    if d.empty:
        raise ValueError(f"User {int(uid)} not found.")
    return d.iloc[0]

def inventory(uid):
    return q("""SELECT product_id AS ID, name AS Product, brand AS Brand,
                category AS Category, open_date AS Opened,
                expiry_date AS Label_expiry, pao_months AS PAO_months,
                qty_left_pct AS Left_pct, usage_per_week AS Uses_per_week,
                storage AS Storage
                FROM products WHERE user_id=? ORDER BY product_id""", (int(uid),))

def profile_data(uid):
    return user_row(uid)

def _vals(name, age, skin, clim, life, con, alg, oth):
    allergies = list(alg or []) + [x.strip() for x in str(oth or "").split(",") if x.strip()]
    return (
        (name or "New user").strip(), int(age), skin,
        ", ".join(allergies) or "None", clim, life,
        ", ".join(con or []) or "None"
    )

def create_profile(name, age, skin, clim, life, con, alg, oth):
    with get_conn() as c:
        uid = c.execute(
            """INSERT INTO users(name,age,skin_type,allergies,climate,lifestyle,concerns)
               VALUES(?,?,?,?,?,?,?)""",
            _vals(name, age, skin, clim, life, con, alg, oth)
        ).lastrowid
    return uid

def update_profile(uid, name, age, skin, clim, life, con, alg, oth):
    user_row(uid)
    with get_conn() as c:
        c.execute(
            """UPDATE users SET name=?,age=?,skin_type=?,allergies=?,
               climate=?,lifestyle=?,concerns=? WHERE user_id=?""",
            (*_vals(name, age, skin, clim, life, con, alg, oth), int(uid))
        )

def add_product(uid, name, brand, cat, open_date, pao, ing, qty, use, storage):
    user_row(uid)
    if not name.strip() or not ing.strip():
        raise ValueError("Product name and ingredients are required.")
    od = pd.Timestamp(open_date)
    exp = (od + pd.DateOffset(months=int(pao))).strftime("%Y-%m-%d")
    with get_conn() as c:
        c.execute(
            """INSERT INTO products
            (user_id,name,brand,category,open_date,expiry_date,pao_months,
             ingredients,qty_left_pct,usage_per_week,storage)
             VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (int(uid), name.strip(), (brand or "").strip(), cat,
             od.strftime("%Y-%m-%d"), exp, int(pao), ing.strip(),
             float(qty), float(use), storage)
        )

def delete_product(uid, pid):
    user_row(uid)
    with get_conn() as c:
        c.execute("DELETE FROM products WHERE product_id=? AND user_id=?",
                  (int(pid), int(uid)))

def score_box(res):
    stype = res["skin_type"]
    return f"""
    <div class="section-card">
    <h3>Skin analysis result</h3>
    <h2>{html.escape(stype)}
    <span style="font-size:14px;color:#777">({res["type_conf"]:.0%} CNN confidence)</span></h2>
    <b>Acne:</b> {res["acne"]:.0f}/100 &nbsp;|&nbsp;
    <b>Pigmentation:</b> {res["pigmentation"]:.0f}/100 &nbsp;|&nbsp;
    <b>Redness:</b> {res["redness"]:.0f}/100 &nbsp;|&nbsp;
    <b>Hydration:</b> {res["hydration"]:.0f}/100 &nbsp;|&nbsp;
    <b>Overall:</b> {res["overall"]:.0f}/100
    </div>
    """

def alerts_tables(uid):
    rk, al = run_user(int(uid), explain_high=True, save=True)
    names = rk.set_index("product_id")["name"]
    A = pd.DataFrame({
        "Severity": al.severity,
        "Type": al.alert_type,
        "Product": al.product_id.map(names),
        "Message": al.message
    })
    R = pd.DataFrame({
        "Priority": rk.priority,
        "Product": rk["name"],
        "Category": rk.category,
        "Days left": rk.days_left.astype(int),
        "Confidence %": rk.confidence.astype(int),
        "Model risk": rk.risk,
        "Waste %": rk.waste_pct.astype(int),
        "Reason": rk.reason
    })
    why = "\n".join(
        f"- **{r['name']}**: {r['why_ml']}"
        for _, r in rk[rk.priority == "High"].iterrows()
        if r["why_ml"]
    ) or "_No High-priority products._"
    return rk, al, A, R, why

def routine_for(uid, use_scan=True, scan=None):
    sc = scan if (use_scan and scan is not None) else (BR.latest_scan(uid) if use_scan else None)
    override = None if sc is None else str(sc["skin_type"])
    R = BR.build_routine(
        uid, use_scan=use_scan, scan=sc,
        skin_override=override, fill=make_fill()
    )
    bad = BR.validate(R, R["user"]["allergies"])
    BR.save_routine(uid, R)
    return R, bad

def display_routine(R):
    st.subheader("☀️ Morning routine")
    am = [x for x in R["items"] if x["slot"] == "AM"]
    for n, it in enumerate(am, 1):
        tag = "✨ Suggested" if it["product_id"] < 0 else "Owned"
        st.markdown(f"**{n}. {it['category']} — {it['name']}**  \n"
                    f"`{tag}` · {it['days']}  \n"
                    f"{it['reason']}")
        if it["product_id"] < 0:
            st.caption("Look for: " + LOOK.get(it["product_id"], "basic compatible ingredients"))

    st.subheader("🌙 Evening routine")
    pm = [x for x in R["items"] if x["slot"] == "PM"]
    for n, it in enumerate(pm, 1):
        tag = "✨ Suggested" if it["product_id"] < 0 else "Owned"
        st.markdown(f"**{n}. {it['category']} — {it['name']}**  \n"
                    f"`{tag}` · {it['days']}  \n"
                    f"{it['reason']}")

    st.subheader("📅 Weekly plan")
    st.dataframe(R["grid"], use_container_width=True)

    if R["excluded"]:
        st.subheader("🚫 Not used, and why")
        st.dataframe(pd.DataFrame(R["excluded"]), use_container_width=True)

    if R["notes"]:
        st.subheader("📝 Notes and safety checks")
        for n in R["notes"]:
            st.write("•", n)

    if bad:
        st.warning("Independent routine safety check found: " + "; ".join(bad))
    else:
        st.success("Independent routine safety check: no rule violations detected.")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## 👤 User")
    user_ids = q("SELECT user_id FROM users ORDER BY user_id")["user_id"].tolist()
    default_uid = 1 if 1 in user_ids else (user_ids[0] if user_ids else 1)
    uid = st.number_input("User ID", min_value=1, value=int(default_uid), step=1)
    st.caption("Demo users from the supplied database are already available.")
    if st.button("🔄 Refresh"):
        st.rerun()

# -------------------- Tabs --------------------
tab_profile, tab_alerts, tab_skin, tab_routine, tab_dash = st.tabs(
    ["👤 Profile & Inventory", "🔔 Alerts & Priority", "📸 Skin Analysis", "🧴 Routine", "📊 Dashboard"]
)

with tab_profile:
    try:
        u = profile_data(uid)
        st.success(f"Loaded: {u['name']} · {u['skin_type']} skin · {u['climate']} climate")
        c1, c2, c3 = st.columns(3)
        c1.metric("Age", int(u["age"]))
        c2.metric("Lifestyle", u["lifestyle"])
        c3.metric("Concerns", len([x for x in str(u["concerns"]).split(",") if x.strip() and x.strip() != "None"]))

        st.markdown("### Current profile")
        st.write(f"**Name:** {u['name']}")
        st.write(f"**Skin type:** {u['skin_type']}")
        st.write(f"**Climate:** {u['climate']}")
        st.write(f"**Lifestyle:** {u['lifestyle']}")
        st.write(f"**Concerns:** {u['concerns']}")
        st.write(f"**Allergies:** {u['allergies']}")

        st.markdown("### 🧴 My cosmetics")
        inv = inventory(uid)
        st.dataframe(inv, use_container_width=True, hide_index=True)

        with st.expander("✏️ Edit profile"):
            name = st.text_input("Name", value=str(u["name"]), key="edit_name")
            age = st.number_input("Age", 16, 80, int(u["age"]), key="edit_age")
            skin = st.selectbox("Profile skin type", SKIN, index=SKIN.index(str(u["skin_type"])) if str(u["skin_type"]) in SKIN else 0)
            clim = st.selectbox("Climate", CLIMATES, index=CLIMATES.index(str(u["climate"])) if str(u["climate"]) in CLIMATES else 0)
            life = st.selectbox("Lifestyle", LIFESTYLES, index=LIFESTYLES.index(str(u["lifestyle"])) if str(u["lifestyle"]) in LIFESTYLES else 0)
            old_con = [x.strip() for x in str(u["concerns"]).split(",") if x.strip() in CONCERNS]
            con = st.multiselect("Concerns", CONCERNS, default=old_con)
            old_alg = [x.strip() for x in str(u["allergies"]).split(",") if x.strip() in ALLERGENS]
            alg = st.multiselect("Known allergies", ALLERGENS, default=old_alg)
            oth = st.text_input("Other allergies (comma-separated)", value="")
            if st.button("Save profile", type="primary"):
                update_profile(uid, name, age, skin, clim, life, con, alg, oth)
                st.success("Profile updated.")
                st.rerun()

        with st.expander("➕ Add product"):
            with st.form("add_product_form"):
                pn = st.text_input("Product name")
                pb = st.text_input("Brand")
                pc = st.selectbox("Category", CATS)
                po = st.date_input("Open date", value=pd.Timestamp.today().date())
                pp = st.slider("PAO (months)", 3, 36, 12)
                pi = st.text_area("Ingredients (comma-separated)")
                pq = st.slider("Amount left (%)", 0, 100, 100)
                pu = st.slider("Uses per week", 0.0, 14.0, 7.0, step=0.5)
                ps = st.selectbox("Storage", STORAGES)
                if st.form_submit_button("Add product", type="primary"):
                    try:
                        add_product(uid, pn, pb, pc, po, pp, pi, pq, pu, ps)
                        st.success("Product added.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        with st.expander("🗑️ Remove product"):
            ids = inv["ID"].tolist() if len(inv) else []
            if ids:
                pid = st.selectbox("Product ID", ids)
                if st.button("Remove selected product"):
                    delete_product(uid, pid)
                    st.success("Product removed.")
                    st.rerun()
            else:
                st.info("No products to remove.")

    except Exception as e:
        st.error(str(e))

with tab_alerts:
    st.subheader("🔔 Smart expiry alerts and product priority")
    try:
        inv = inventory(uid)
        if len(inv) == 0:
            st.info("No products yet. Add products in Profile & Inventory.")
        else:
            if st.button("Analyze my inventory", type="primary"):
                with st.spinner("Running shelf-life models and SHAP explanations..."):
                    rk, al, A, R, why = alerts_tables(uid)
                st.session_state["rk"] = rk
                st.session_state["al"] = al
                st.session_state["A"] = A
                st.session_state["R"] = R
                st.session_state["why"] = why
            if "A" in st.session_state:
                st.dataframe(st.session_state["A"], use_container_width=True, hide_index=True)
                st.subheader("Product priority ranking")
                st.dataframe(st.session_state["R"], use_container_width=True, hide_index=True)
                st.subheader("Why the model thinks so")
                st.markdown(st.session_state["why"])
            st.divider()
            st.subheader("Global SHAP explanations")
            c1, c2 = st.columns(2)
            for c, f in zip([c1, c2], ["shap_summary.png", "shap_bar.png"]):
                p = os.path.join(P["outputs"], f)
                if os.path.exists(p):
                    c.image(p, use_container_width=True)

    except Exception as e:
        st.error(f"Inventory analysis failed: {e}")

with tab_skin:
    st.subheader("📸 AI-powered facial skin analysis")
    st.caption("Front-facing photo + good lighting gives the most reliable result.")
    img_file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input("Or take a photo")
    selected = camera_file if camera_file is not None else img_file
    save_it = st.checkbox("Save this scan to my history", value=True)

    if st.button("Analyze skin", type="primary"):
        if selected is None:
            st.error("Please upload or capture a face photo first.")
        else:
            try:
                img = Image.open(selected).convert("RGB")
                arr = np.array(img)
                with st.spinner("Running face analysis, CNNs and Grad-CAM..."):
                    res = analyze(arr)
                    if save_it:
                        save_scan(int(uid), res, arr)
                    st.session_state["scan_result"] = res
                if "no face" in res["face_method"]:
                    st.warning("No face detected. The whole image was used, so the result has lower reliability.")
            except Exception as e:
                st.error(f"Skin analysis failed: {e}")

    if "scan_result" in st.session_state:
        res = st.session_state["scan_result"]
        st.markdown(score_box(res), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.image(res["imgs"]["face"], caption="Skin region used", use_container_width=True)
        c2.image(res["imgs"]["gc_type"], caption="Grad-CAM: skin type", use_container_width=True)
        c3.image(res["imgs"]["gc_acne"], caption="Grad-CAM: acne", use_container_width=True)

        st.subheader("🔍 Why these results?")
        for w in res["why"]:
            st.write("•", w)
        st.info("Cosmetic guidance only, not a medical diagnosis. Skin type/acne use CNNs trained on small public datasets; pigmentation, redness and hydration are image-analysis estimates.")

        if st.button("✨ Build my routine from this scan", type="primary"):
            try:
                R, bad = routine_for(uid, use_scan=True, scan=res)
                st.session_state["routine"] = R
                st.session_state["routine_bad"] = bad
                st.success("Personalised routine created. Open the Routine tab.")
            except Exception as e:
                st.error(f"Routine generation failed: {e}")

with tab_routine:
    st.subheader("🧴 AI Beauty Routine Optimizer")
    use_scan = st.checkbox("Use my latest skin scan", value=True)
    if st.button("Generate my routine", type="primary"):
        try:
            with st.spinner("Building a rule-checked routine..."):
                R, bad = routine_for(uid, use_scan=use_scan)
            st.session_state["routine"] = R
            st.session_state["routine_bad"] = bad
        except Exception as e:
            st.error(f"Routine generation failed: {e}")

    if "routine" in st.session_state:
        R = st.session_state["routine"]
        st.success(R["summary"])
        display_routine(R)

with tab_dash:
    st.subheader("📊 BeautyMind AI Dashboard")
    try:
        inv = inventory(uid)
        if len(inv):
            if st.button("Refresh dashboard", type="primary"):
                with st.spinner("Preparing dashboard..."):
                    rk, al = run_user(int(uid), explain_high=False, save=True)
                    sc = D.get_scans(uid)
                    st.session_state["dash"] = (rk, al, sc)

            if "dash" not in st.session_state:
                rk, al = run_user(int(uid), explain_high=False, save=True)
                sc = D.get_scans(uid)
            else:
                rk, al, sc = st.session_state["dash"]

            st.markdown(D.kpi_html(rk, al, uid), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.plotly_chart(D.fig_status(rk), use_container_width=True)
            c2.plotly_chart(D.fig_days(rk), use_container_width=True)
            c3, c4 = st.columns(2)
            c3.plotly_chart(D.fig_waste(rk), use_container_width=True)
            c4.plotly_chart(D.fig_alerts(al), use_container_width=True)
            st.plotly_chart(D.fig_trend(sc), use_container_width=True)
            st.markdown(D.trend_md(sc))

            st.divider()
            if st.button("➕ Add synthetic demo skin history"):
                D.seed_demo_scans(uid)
                st.success("Added synthetic demo history. Refresh the dashboard.")
                st.rerun()
        else:
            st.info("Add at least one cosmetic product to activate the inventory dashboard.")
    except Exception as e:
        st.error(f"Dashboard failed: {e}")

st.markdown("---")
st.caption("BeautyMind AI research prototype • Shelf-life data is synthetic • Skin analysis is cosmetic guidance, not medical advice • Routine rules are editable heuristics.")

import json, os, numpy as np, cv2, tensorflow as tf
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from bm_config import P, get_conn

SZ = 224
CAL = dict(red_delta=4.0, red_full=.20, pig_delta=6.0, pig_full=.10, rough_full=4.0)   # heuristic constants, tune on real photos
LV = lambda s: "low" if s < 33 else "moderate" if s < 66 else "high"
try: import mediapipe as mp; _FM = mp.solutions.face_mesh
except Exception: _FM = None
_M, _G = {}, {}

def load_cnn(name):
    if name not in _M: _M[name] = (tf.keras.models.load_model(f"{P['models']}/skin_{name}.keras"), json.load(open(f"{P['models']}/skin_{name}_classes.json")))
    return _M[name]

def _hull(lm, edges): return cv2.convexHull(lm[sorted({i for e in edges for i in e})].astype(np.int32))

def skin_mask(rgb):                                   # skin only: face oval minus eyes, brows, lips
    h, w = rgb.shape[:2]; m = np.zeros((h, w), np.uint8)
    if _FM:
        with _FM.FaceMesh(static_image_mode=True, max_num_faces=1) as fm: r = fm.process(rgb)
        if r.multi_face_landmarks:
            lm = np.array([[p.x*w, p.y*h] for p in r.multi_face_landmarks[0].landmark])
            cv2.fillConvexPoly(m, _hull(lm, _FM.FACEMESH_FACE_OVAL), 255); ex = np.zeros_like(m)
            for s in (_FM.FACEMESH_LIPS, _FM.FACEMESH_LEFT_EYE, _FM.FACEMESH_RIGHT_EYE, _FM.FACEMESH_LEFT_EYEBROW, _FM.FACEMESH_RIGHT_EYEBROW):
                cv2.fillConvexPoly(ex, _hull(lm, s), 255)
            k = (max(3, w//60)) | 1
            m[cv2.dilate(ex, np.ones((k, k), np.uint8)) > 0] = 0
            return cv2.erode(m, np.ones((k, k), np.uint8)), "MediaPipe FaceMesh"
    # Removed the problematic cv2.CascadeClassifier fallback
    m[:] = 255; return m, "no face found - whole image used, low reliability"

def crop_face(rgb, m, pad=.08):
    ys, xs = np.where(m > 0); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); py, px = int((y1-y0)*pad), int((x1-x0)*pad)
    y0, y1, x0, x1 = max(0, y0-py), min(m.shape[0], y1+py), max(0, x0-px), min(m.shape[1], x1+px)
    return rgb[y0:y1, x0:x1], m[y0:y1, x0:x1]

def _mblur(x, mk, s): return cv2.GaussianBlur(x*mk, (0, 0), s) / (cv2.GaussianBlur(mk, (0, 0), s) + 1e-6)

def color_scores(crop, mk):                           # local deviation from surrounding skin, in Lab colour space
    c = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA); sk = cv2.resize(mk, (256, 256), interpolation=cv2.INTER_NEAREST) > 0
    if sk.sum() < 2000: sk[:] = True
    lab = cv2.cvtColor(c, cv2.COLOR_RGB2LAB).astype(np.float32); Lc, A = lab[..., 0]*100/255, lab[..., 1] - 128; mf = sk.astype(np.float32)
    dL = _mblur(Lc, mf, 20) - cv2.GaussianBlur(Lc, (0, 0), 2); dA = cv2.GaussianBlur(A, (0, 0), 2) - _mblur(A, mf, 20)
    pig = np.clip(np.mean(dL[sk] > CAL["pig_delta"]) / CAL["pig_full"] * 100, 0, 100)
    red = np.clip(np.mean(dA[sk] > CAL["red_delta"]) / CAL["red_full"] * 100, 0, 100)
    rough = np.std((Lc - cv2.GaussianBlur(Lc, (0, 0), 1.5))[sk])
    return float(pig), float(red), float(np.clip(rough / CAL["rough_full"], 0, 1))

def gradcam(name, x):                                 # x: preprocessed (1,224,224,3); last conv layer = out_relu
    m, _ = load_cnn(name)
    if name not in _G: _G[name] = tf.keras.Model(m.input, [m.get_layer("out_relu").output, m.output])
    with tf.GradientTape() as t:
        conv, pred = _G[name](tf.convert_to_tensor(x)); cls = int(tf.argmax(pred[0])); s = pred[:, cls]
    w = tf.reduce_mean(t.gradient(s, conv), axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], -1))[0].numpy()
    return cam / (cam.max() + 1e-8), pred[0].numpy(), cls

def overlay(img, cam, a=.45):
    hm = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, img.shape[1::-1])), cv2.COLORMAP_JET)[..., ::-1]
    return cv2.addWeighted(img, 1 - a, hm, a, 0)

def focus(cam):                                       # where the CNN looked, in words (left/right as seen in the photo)
    c = cv2.resize(cam, (90, 90)); r = {"forehead": c[:30].sum(), "left cheek": c[30:60, :30].sum(), "nose area": c[30:60, 30:60].sum(),
                                        "right cheek": c[30:60, 60:].sum(), "chin/jaw": c[60:].sum()}
    t = sum(r.values()) + 1e-8; top = [k for k in sorted(r, key=r.get, reverse=True)[:2] if r[k]/t >= .25]
    return " and ".join(top) or "the face broadly"

def analyze(rgb):
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8); m, how = skin_mask(rgb); crop, mk = crop_face(rgb, m)
    img = cv2.resize(crop, (SZ, SZ), interpolation=cv2.INTER_AREA); x = preprocess_input(img[None].astype("float32"))
    cam_t, p_t, i_t = gradcam("type", x); cam_a, p_a, _ = gradcam("acne", x)
    tn = [n.lower() for n in load_cnn("type")[1]]; pt = dict(zip(tn, p_t.astype(float)))
    stype = "Combination" if min(pt.get("dry", 0), pt.get("oily", 0)) >= .30 else tn[i_t].capitalize()   # heuristic: strong dry+oily signals
    acne = float(p_a @ np.arange(len(p_a)) / (len(p_a) - 1) * 100)      # expected severity over ordered acne grades
    pig, red, rough = color_scores(crop, mk)
    hyd = float(np.clip(100 * (1 - (.7 * pt.get("dry", 0) + .3 * rough)), 0, 100))
    overall = float(np.mean([100 - acne, 100 - pig, 100 - red, hyd]))
    why = [f"Skin type {stype}: type-CNN probabilities " + ", ".join(f"{k} {v:.0%}" for k, v in pt.items()) + f"; it focused mostly on the {focus(cam_t)}."
           + (" Dry and oily signals are both strong, so it is labelled Combination." if stype == "Combination" else ""),
           f"Acne {acne:.0f}/100 ({LV(acne)}): expected severity from the acne-grading CNN; it focused mostly on the {focus(cam_a)}.",
           f"Pigmentation {pig:.0f}/100 ({LV(pig)}): share of skin darker than its surroundings (image analysis, not a CNN).",
           f"Redness {red:.0f}/100 ({LV(red)}): share of skin redder than its surroundings (image analysis, not a CNN).",
           f"Hydration {hyd:.0f}/100 (higher is better): proxy from dryness probability ({pt.get('dry', 0):.0%}) and fine texture, not a measurement."]
    vis = rgb.copy(); vis[m == 0] = (vis[m == 0] * .25).astype(np.uint8)
    return dict(skin_type=stype, type_conf=float(p_t[i_t]), probs=pt, acne=round(acne, 1), pigmentation=round(pig, 1), redness=round(red, 1),
                hydration=round(hyd, 1), overall=round(overall, 1), face_method=how, why=why,
                imgs=dict(face=vis, gc_type=overlay(img, cam_t), gc_acne=overlay(img, cam_a)))

def save_scan(user_id, res, rgb=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); path = None
    if rgb is not None:
        os.makedirs(f"{P['outputs']}/scans", exist_ok=True)
        path = f"{P['outputs']}/scans/u{user_id}_{ts.replace(':', '').replace(' ', '_').replace('-', '')}.jpg"; cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    with get_conn() as c:
        c.execute("INSERT INTO skin_scans(user_id,scan_date,skin_type,acne,pigmentation,redness,hydration,overall,image_path) VALUES(?,?,?,?,?,?,?,?,?)",
                  (int(user_id), ts, res["skin_type"], res["acne"], res["pigmentation"], res["redness"], res["hydration"], res["overall"], path))


# ---- patched: OpenCV 5 has no CascadeClassifier, so the Haar fallback is guarded
def skin_mask(rgb):
    h, w = rgb.shape[:2]; m = np.zeros((h, w), np.uint8)
    if _FM:
        with _FM.FaceMesh(static_image_mode=True, max_num_faces=1) as fm: r = fm.process(rgb)
        if r.multi_face_landmarks:
            lm = np.array([[p.x*w, p.y*h] for p in r.multi_face_landmarks[0].landmark])
            cv2.fillConvexPoly(m, _hull(lm, _FM.FACEMESH_FACE_OVAL), 255); ex = np.zeros_like(m)
            for s in (_FM.FACEMESH_LIPS, _FM.FACEMESH_LEFT_EYE, _FM.FACEMESH_RIGHT_EYE, _FM.FACEMESH_LEFT_EYEBROW, _FM.FACEMESH_RIGHT_EYEBROW):
                cv2.fillConvexPoly(ex, _hull(lm, s), 255)
            k = (max(3, w//60)) | 1
            m[cv2.dilate(ex, np.ones((k, k), np.uint8)) > 0] = 0
            return cv2.erode(m, np.ones((k, k), np.uint8)), "MediaPipe FaceMesh"
    try:
        f = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 5)
    except Exception: f = []
    if len(f):
        x, y, bw, bh = max(f, key=lambda b: b[2]*b[3])
        cv2.ellipse(m, (x+bw//2, y+bh//2), (int(bw*.38), int(bh*.48)), 0, 0, 360, 255, -1); return m, "OpenCV Haar (coarse)"
    m[:] = 255; return m, "no face found - whole image used, low reliability"

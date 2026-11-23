"""
NE111 Term Project - Plot Digitizer
A single-page Streamlit app for extracting x-y data from an image of a graph.

Workflow
  1. Upload an image (png / jpg / gif / tiff / bmp, colour or greyscale).
  2. Calibrate: click the origin, a point on the x-axis and a point on the y-axis,
     and type in the known axis values of those points.
  3. Pick a dataset and click on its data points.
  4. Download everything as a CSV.

How calibration handles rotated scans
  The three calibration clicks define two pixel vectors (origin->x-point and
  origin->y-point). Any clicked pixel is written as a combination of those two
  vectors, so the axes do NOT have to be horizontal/vertical in the image.
  (Linear axes are mapped directly; log axes are mapped in log10 space.)
"""

import csv
import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]
CAL_COLOR = (255, 0, 255)  # magenta crosshairs for calibration points
CAL_KEYS = ["O", "X", "Y"]
CAL_LABELS = {"O": "Origin", "X": "x-axis point", "Y": "y-axis point"}
MODES = [
    "1. Set origin",
    "2. Set point on x-axis",
    "3. Set point on y-axis",
    "4. Add data points",
]


# ----------------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------------
def load_image(uploaded):
    """Open an uploaded file as an RGB PIL image (handles greyscale, 16-bit,
    palette, transparency, multi-frame gif/tiff and EXIF orientation)."""
    img = Image.open(uploaded)
    try:
        img.seek(0)  # first frame of gif / tiff
    except Exception:
        pass
    img = ImageOps.exif_transpose(img)

    if img.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        arr = np.asarray(img, dtype=float)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8))

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        img = bg

    return img.convert("RGB")


# ----------------------------------------------------------------------------
# Calibration maths
# ----------------------------------------------------------------------------
def _fwd(value, is_log):
    return math.log10(value) if is_log else value


def _inv(value, is_log):
    if not is_log:
        return value
    try:
        return 10.0 ** value
    except OverflowError:
        return float("nan")


class Calibration:
    """Maps between image pixels and data values using three reference points."""

    def __init__(self, p0, px, py, x0, y0, x1, y1, xlog, ylog):
        self.p0 = np.array(p0, dtype=float)
        self.ex = np.array(px, dtype=float) - self.p0  # pixel vector for x0 -> x1
        self.ey = np.array(py, dtype=float) - self.p0  # pixel vector for y0 -> y1
        self.u0, self.u1 = _fwd(x0, xlog), _fwd(x1, xlog)
        self.v0, self.v1 = _fwd(y0, ylog), _fwd(y1, ylog)
        self.xlog, self.ylog = xlog, ylog
        self.m = np.column_stack([self.ex, self.ey])

    def pix_to_data(self, px, py):
        s, t = np.linalg.solve(self.m, np.array([px, py], dtype=float) - self.p0)
        u = self.u0 + s * (self.u1 - self.u0)
        v = self.v0 + t * (self.v1 - self.v0)
        return _inv(u, self.xlog), _inv(v, self.ylog)

    def data_to_pix(self, x, y):
        """Returns (px, py) or None if the value is not representable (e.g. log of <= 0)."""
        if (self.xlog and x <= 0) or (self.ylog and y <= 0):
            return None
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        s = (_fwd(x, self.xlog) - self.u0) / (self.u1 - self.u0)
        t = (_fwd(y, self.ylog) - self.v0) / (self.v1 - self.v0)
        p = self.p0 + s * self.ex + t * self.ey
        return float(p[0]), float(p[1])


def make_calibration(pix, x0, y0, x1, y1, xlog, ylog):
    """Returns (Calibration or None, message). message explains what is wrong."""
    if any(pix[k] is None for k in CAL_KEYS):
        missing = [CAL_LABELS[k] for k in CAL_KEYS if pix[k] is None]
        return None, "Still to click: " + ", ".join(missing) + "."
    if xlog and (x0 <= 0 or x1 <= 0):
        return None, "Log x-axis: the origin x value and the x-axis point value must both be > 0."
    if ylog and (y0 <= 0 or y1 <= 0):
        return None, "Log y-axis: the origin y value and the y-axis point value must both be > 0."
    if _fwd(x0, xlog) == _fwd(x1, xlog):
        return None, "The two x values used for calibration must be different."
    if _fwd(y0, ylog) == _fwd(y1, ylog):
        return None, "The two y values used for calibration must be different."

    p0 = np.array(pix["O"], dtype=float)
    ex = np.array(pix["X"], dtype=float) - p0
    ey = np.array(pix["Y"], dtype=float) - p0
    det = ex[0] * ey[1] - ex[1] * ey[0]
    if abs(det) < 1e-3 * max(np.linalg.norm(ex) * np.linalg.norm(ey), 1e-12):
        return None, "The three calibration points are (nearly) in a line - re-click them."
    return Calibration(pix["O"], pix["X"], pix["Y"], x0, y0, x1, y1, xlog, ylog), ""


# ----------------------------------------------------------------------------
# CSV export
# ----------------------------------------------------------------------------
def _fmt(v):
    return format(float(v), ".10g")


def build_csv(datasets, x_label, y_label):
    """Row 1: dataset names, row 2: axis labels, then x,y values."""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    if not datasets:
        return ""
    w.writerow(sum(([d["name"], ""] for d in datasets), []))
    w.writerow([x_label, y_label] * len(datasets))
    n = max((len(d["points"]) for d in datasets), default=0)
    for i in range(n):
        row = []
        for d in datasets:
            if i < len(d["points"]):
                x, y = d["points"][i]
                row += [_fmt(x), _fmt(y)]
            else:
                row += ["", ""]
        w.writerow(row)
    return out.getvalue()


# ----------------------------------------------------------------------------
# Session state helpers
# ----------------------------------------------------------------------------
def new_dataset():
    i = st.session_state.next_id
    st.session_state.next_id += 1
    ds = {
        "id": i,
        "name": f"Dataset {i}",
        "color": PALETTE[(i - 1) % len(PALETTE)],
        "points": [],  # list of [x, y] in DATA units
        "ver": 0,      # bumped whenever points change so the table editor resets
    }
    st.session_state.datasets.append(ds)
    st.session_state["active_sel"] = i


def get_ds(ds_id):
    for d in st.session_state.datasets:
        if d["id"] == ds_id:
            return d
    return None


def undo_point():
    ds = get_ds(st.session_state.get("active_sel"))
    if ds and ds["points"]:
        ds["points"].pop()
        ds["ver"] += 1


def delete_active():
    aid = st.session_state.get("active_sel")
    st.session_state.datasets = [d for d in st.session_state.datasets if d["id"] != aid]
    rest = st.session_state.datasets
    st.session_state["active_sel"] = rest[0]["id"] if rest else None


def reset_calibration():
    st.session_state.cal = {k: None for k in CAL_KEYS}


def init_state():
    ss = st.session_state
    ss.setdefault("datasets", [])
    ss.setdefault("next_id", 1)
    ss.setdefault("cal", {k: None for k in CAL_KEYS})
    ss.setdefault("cal_angle", None)
    ss.setdefault("file_key", None)
    ss.setdefault("base_img", None)
    ss.setdefault("rot_cache", None)
    ss.setdefault("last_click", None)
    ss.setdefault("flash", None)
    if not ss.datasets:
        new_dataset()


def reset_for_new_image():
    st.session_state.datasets = []
    st.session_state.next_id = 1
    st.session_state.cal_angle = None
    reset_calibration()
    new_dataset()


def get_rotated(base, angle):
    key = (st.session_state.file_key, angle)
    cache = st.session_state.rot_cache
    if cache and cache[0] == key:
        return cache[1]
    if angle % 360 == 0:
        img = base
    else:
        img = base.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    st.session_state.rot_cache = (key, img)
    return img


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def render_view(img, box, display_w, cal_pix, datasets, cal):
    """Crop to `box`, resize to display width, draw calibration + data markers."""
    l, t, r, b = box
    crop = img.crop(box)
    display_h = max(1, round(display_w * (b - t) / (r - l)))
    resample = Image.LANCZOS if display_w < (r - l) else Image.BICUBIC
    view = crop.resize((display_w, display_h), resample)
    draw = ImageDraw.Draw(view)
    sx, sy = display_w / (r - l), display_h / (b - t)

    def to_disp(px, py):
        return (px - l) * sx, (py - t) * sy

    # calibration markers
    for k in CAL_KEYS:
        p = cal_pix.get(k)
        if p is None:
            continue
        x, y = to_disp(*p)
        if -20 <= x <= display_w + 20 and -20 <= y <= display_h + 20:
            draw.line([(x - 10, y), (x + 10, y)], fill=CAL_COLOR, width=2)
            draw.line([(x, y - 10), (x, y + 10)], fill=CAL_COLOR, width=2)
            draw.text((x + 8, y + 6), k, fill=CAL_COLOR)

    # data markers
    if cal is not None:
        for d in datasets:
            for x_val, y_val in d["points"]:
                p = cal.data_to_pix(x_val, y_val)
                if p is None:
                    continue
                x, y = to_disp(*p)
                if -10 <= x <= display_w + 10 and -10 <= y <= display_h + 10:
                    rad = 5
                    draw.ellipse([x - rad - 1, y - rad - 1, x + rad + 1, y + rad + 1], fill="white")
                    draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=d["color"], outline="black")
    return view


def same_points(a, b):
    if len(a) != len(b):
        return False
    if not a:
        return True
    return bool(np.allclose(np.array(a, float), np.array(b, float), rtol=1e-12, atol=1e-12))


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Plot Digitizer", page_icon="📈", layout="wide")
    init_state()
    ss = st.session_state

    st.title("📈 Plot Digitizer")

    # ---------------- Sidebar: image ----------------
    with st.sidebar:
        st.header("1. Image")
        up = st.file_uploader(
            "Upload a graph image",
            type=["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp"],
        )

    if up is None:
        st.info("Upload an image of a graph in the sidebar to get started.")
        st.markdown(
            """
**How it works**
1. Upload a picture of a graph (colour or greyscale; png, jpg, gif, tiff).
2. *Calibrate:* click the **origin**, a point on the **x-axis** and a point on the **y-axis**,
   then type the real axis values of those points in the sidebar. Choose linear or log scales.
3. Create/select a **dataset** and click on its data points. Each dataset has its own colour.
4. Zoom in for precision, edit points in the table if needed, then **download the CSV**.

If your scan is tilted, you don't need to fix it first - the calibration handles rotation -
but the rotation sliders can straighten it to make clicking easier.
"""
        )
        return

    fk = (up.name, up.size)
    if ss.file_key != fk:
        try:
            ss.base_img = load_image(up)
        except Exception as exc:  # unreadable image
            st.error(f"Could not read that image: {exc}")
            return
        ss.file_key = fk
        ss.rot_cache = None
        reset_for_new_image()

    # ---------------- Sidebar: rest of controls ----------------
    with st.sidebar:
        coarse = st.selectbox("Rotate image", [0, 90, 180, 270], format_func=lambda a: f"{a}°")
        fine = st.slider("Fine rotation (° counter-clockwise)", -45.0, 45.0, 0.0, 0.1)
        angle = float(coarse) + float(fine)
        if ss.cal_angle != angle:
            if ss.cal_angle is not None:
                reset_calibration()  # pixel positions change when the image rotates
            ss.cal_angle = angle
        st.caption("Changing the rotation clears the calibration clicks (data points are kept).")

        st.header("2. Axes")
        x_label = st.text_input("x-axis label", value="x")
        y_label = st.text_input("y-axis label", value="y")
        c1, c2 = st.columns(2)
        xscale = c1.radio("x scale", ["Linear", "Log"])
        yscale = c2.radio("y scale", ["Linear", "Log"])
        xlog, ylog = xscale == "Log", yscale == "Log"

        st.header("3. Calibration")
        st.caption("Enter the true values at the three points you click on the image.")
        o1, o2 = st.columns(2)
        x0 = o1.number_input("Origin x", value=1.0 if xlog else 0.0, format="%g", key=f"x0_{xscale}")
        y0 = o2.number_input("Origin y", value=1.0 if ylog else 0.0, format="%g", key=f"y0_{yscale}")
        x1 = st.number_input(
            "x value at the x-axis point", value=10.0 if xlog else 1.0, format="%g", key=f"x1_{xscale}"
        )
        y1 = st.number_input(
            "y value at the y-axis point", value=10.0 if ylog else 1.0, format="%g", key=f"y1_{yscale}"
        )
        st.caption(
            "The 'origin' can be any point with known x and y (it need not be 0,0 - "
            "required for log axes). The x-axis point should share the origin's y value; "
            "the y-axis point should share its x value."
        )
        for k in CAL_KEYS:
            st.write(("✅ " if ss.cal[k] else "⬜ ") + CAL_LABELS[k])
        st.button("Reset calibration", on_click=reset_calibration)

        st.header("4. Datasets")
        ds_list = ss.datasets
        if ds_list:
            names = {d["id"]: d["name"] for d in ds_list}
            ids = [d["id"] for d in ds_list]
            st.selectbox(
                "Active dataset (clicks are added here)",
                ids,
                format_func=lambda i: names.get(i, str(i)),
                key="active_sel",
            )
            active = get_ds(ss.get("active_sel"))
            if active:
                active["name"] = st.text_input(
                    "Dataset name", value=active["name"], key=f"name_{active['id']}"
                )
                st.markdown(
                    f"<span style='color:{active['color']};font-size:1.3em'>●</span> "
                    f"{len(active['points'])} point(s)",
                    unsafe_allow_html=True,
                )
        else:
            active = None
            st.info("No datasets. Add one to start clicking.")
        b1, b2, b3 = st.columns(3)
        b1.button("➕ Add", on_click=new_dataset, help="Add a new dataset")
        b2.button("↩ Undo", on_click=undo_point, help="Remove the last point of the active dataset")
        b3.button("🗑 Delete", on_click=delete_active, help="Delete the active dataset")

    # ---------------- Calibration object ----------------
    cal, cal_msg = make_calibration(ss.cal, x0, y0, x1, y1, xlog, ylog)

    # ---------------- Main: controls above image ----------------
    if ss.flash:
        st.warning(ss.flash)
        ss.flash = None

    first_missing = next((i for i, k in enumerate(CAL_KEYS) if ss.cal[k] is None), 3)
    mode = st.radio("Click mode", MODES, index=first_missing, horizontal=True)
    if mode != MODES[3]:
        st.caption("Click the matching spot on the image. Zoom in for accuracy.")
    elif cal is None:
        st.caption("⚠️ " + cal_msg)
    elif active is None:
        st.caption("⚠️ Add a dataset in the sidebar first.")
    else:
        st.caption(f"Adding points to **{active['name']}**.")

    img = get_rotated(ss.base_img, angle)
    W, H = img.size

    z1, z2, z3, z4 = st.columns(4)
    zoom = z1.slider("Zoom", 1.0, 10.0, 1.0, 0.5)
    pan_x = z2.slider("Pan left ↔ right", 0, 100, 50, disabled=zoom == 1.0)
    pan_y = z3.slider("Pan up ↕ down", 0, 100, 50, disabled=zoom == 1.0)
    display_w = z4.slider("Display width (px)", 400, 1600, 900, 50)

    cw, ch = W / zoom, H / zoom
    left = (W - cw) * pan_x / 100.0
    top = (H - ch) * pan_y / 100.0
    box = (round(left), round(top), min(W, round(left + cw)), min(H, round(top + ch)))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        box = (0, 0, W, H)

    view = render_view(img, box, display_w, ss.cal, ss.datasets, cal)
    value = streamlit_image_coordinates(view, width=display_w, key="digitizer_img")

    # ---------------- Handle a click ----------------
    if value and value.get("width"):
        sig = (value.get("unix_time"), value["x"], value["y"])
        if sig != ss.last_click:
            ss.last_click = sig
            l, t, r, b = box
            px = l + value["x"] / value["width"] * (r - l)
            py = t + value["y"] / value["height"] * (b - t)
            if mode == MODES[0]:
                ss.cal["O"] = (px, py)
            elif mode == MODES[1]:
                ss.cal["X"] = (px, py)
            elif mode == MODES[2]:
                ss.cal["Y"] = (px, py)
            else:
                if cal is None:
                    ss.flash = "Finish the calibration first. " + cal_msg
                elif active is None:
                    ss.flash = "Add a dataset in the sidebar before clicking points."
                else:
                    xv, yv = cal.pix_to_data(px, py)
                    if np.isfinite(xv) and np.isfinite(yv):
                        active["points"].append([float(xv), float(yv)])
                        active["ver"] += 1
            st.rerun()

    # ---------------- Tables (editable) ----------------
    st.subheader("Digitized data")
    if not ss.datasets:
        st.write("No datasets yet.")
    else:
        tabs = st.tabs([d["name"] or f"Dataset {d['id']}" for d in ss.datasets])
        for tab, d in zip(tabs, ss.datasets):
            with tab:
                df = pd.DataFrame(d["points"], columns=["x", "y"], dtype=float)
                edited = st.data_editor(
                    df,
                    key=f"editor_{d['id']}_{d['ver']}",
                    num_rows="dynamic",
                    hide_index=True,
                    column_config={
                        "x": st.column_config.NumberColumn(x_label or "x", format="%.6g"),
                        "y": st.column_config.NumberColumn(y_label or "y", format="%.6g"),
                    },
                )
                new_pts = edited.dropna().astype(float).values.tolist()
                if not same_points(new_pts, d["points"]):
                    d["points"] = new_pts
                    d["ver"] += 1
                    st.rerun()
        st.caption("Edit values or delete rows in the table; the marks on the image update automatically.")

    # ---------------- Download ----------------
    csv_text = build_csv(ss.datasets, x_label, y_label)
    st.download_button(
        "⬇️ Download CSV",
        data=csv_text,
        file_name="digitized_data.csv",
        mime="text/csv",
        disabled=not ss.datasets,
    )


if __name__ == "__main__":
    main()

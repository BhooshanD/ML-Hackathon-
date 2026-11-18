import os
import time

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PathSnap", page_icon="🛣️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.10);
        border-radius: 12px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- API helpers ----------
def api_get(base, path, timeout=120):
    start = time.perf_counter()
    r = requests.get(f"{base}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - start) * 1000


@st.cache_data(ttl=30, show_spinner=False)
def fetch_trips(base):
    data, _ = api_get(base, "/trips", timeout=10)
    return data["trips"]


def build_map(res, tiles):
    gps = res["gps"]
    lats = [p[0] for p in gps]
    lons = [p[1] for p in gps]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)],
                   zoom_start=15, tiles=tiles, control_scale=True)

    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    for i, p in enumerate(gps):
        folium.CircleMarker(p, radius=3, color="#1f77b4", fill=True,
                            fill_opacity=0.8,
                            tooltip=f"GPS point {i}").add_to(gps_layer)
    gps_layer.add_to(m)

    route_layer = folium.FeatureGroup(name="Predicted route", show=True)
    folium.GeoJson(
        res["predicted_route"],
        style_function=lambda f: {"color": "#e63946", "weight": 5, "opacity": 0.9},
        tooltip="Predicted route",
    ).add_to(route_layer)
    route_layer.add_to(m)

    folium.Marker(gps[0], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(gps[-1], tooltip="End",
                  icon=folium.Icon(color="darkred", icon="stop")).add_to(m)

    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(30, 30))

    legend = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; color: #222; padding: 10px 14px;
                border-radius: 8px; border: 1px solid #999; font-size: 13px;">
      <b>Legend</b><br>
      <span style="color:#1f77b4;">&#9679;</span> GPS observations<br>
      <span style="color:#e63946;">&#9644;</span> Predicted route<br>
      <span style="color:green;">&#9679;</span> Start &nbsp;
      <span style="color:darkred;">&#9679;</span> End
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------- Sidebar ----------
st.sidebar.title("🛣️ PathSnap")
st.sidebar.caption("Map-matching & route reconstruction")
api_url = st.sidebar.text_input("API URL", DEFAULT_API).rstrip("/")
tiles = st.sidebar.selectbox(
    "Map style", ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"])

try:
    trips = fetch_trips(api_url)
    st.sidebar.success(f"API connected · {len(trips)} trips")
except Exception as e:
    st.sidebar.error("Cannot reach the API")
    st.error(f"Could not load trips from `{api_url}`.\n\n{e}")
    st.info("Start the API on Laptop 1 and check the URL, firewall and Wi-Fi.")
    st.stop()

if st.sidebar.button("Refresh trip list"):
    fetch_trips.clear()
    st.rerun()

st.session_state.setdefault("history", [])
st.session_state.setdefault("result", None)

# ---------- Header ----------
st.title("PathSnap Evaluation Dashboard")
st.caption("Select a trip, run it through the API, and inspect the reconstructed route.")

tab_run, tab_all, tab_hist = st.tabs(
    ["🔍 Single trip", "📊 Run all trips", "🕘 History"])

# ---------- Single trip ----------
with tab_run:
    c1, c2 = st.columns([3, 1])
    trip_id = c1.selectbox("Evaluation trip", trips)
    run = c2.button("▶ Run prediction", type="primary")

    if run:
        try:
            with st.spinner(f"Running {trip_id} through the pipeline..."):
                res, ms = api_get(api_url, f"/predict/{trip_id}")
            res["_latency_ms"] = ms
            st.session_state["result"] = res
            st.session_state["history"].append({
                "time": time.strftime("%H:%M:%S"),
                "trip": trip_id,
                "accuracy": res.get("accuracy"),
                "latency_ms": round(ms),
                "gps_points": len(res.get("gps", [])),
            })
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    res = st.session_state["result"]
    if res:
        st.subheader(f"Results for {res.get('trip_id', '')}")
        acc = res.get("accuracy")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.2%}" if acc is not None else "N/A")
        m2.metric("GPS points", len(res["gps"]))
        m3.metric("Route vertices",
                  len(res["predicted_route"].get("coordinates", [])))
        m4.metric("Response time", f"{res['_latency_ms']:.0f} ms")
        if acc is not None:
            st.progress(min(max(float(acc), 0.0), 1.0))

        left, right = st.columns([3, 1])
        with left:
            st_folium(build_map(res, tiles), height=560, use_container_width=True,
                      returned_objects=[], key=f"map_{res.get('trip_id')}")
        with right:
            st.markdown("**GPS sample**")
            st.dataframe(pd.DataFrame(res["gps"], columns=["lat", "lon"]).head(15),
                         height=300)
            with st.expander("Raw API response"):
                st.json({k: v for k, v in res.items() if k != "_latency_ms"})
    else:
        st.info("Pick a trip and click **Run prediction**.")

# ---------- All trips ----------
with tab_all:
    st.write("Runs every evaluation trip through the API, one by one.")
    if st.button("▶ Run all trips"):
        rows, bar = [], st.progress(0.0)
        for i, t in enumerate(trips):
            try:
                r, ms = api_get(api_url, f"/predict/{t}")
                rows.append({"trip": t, "accuracy": r.get("accuracy"),
                             "latency_ms": round(ms), "status": "OK"})
            except Exception as e:
                rows.append({"trip": t, "accuracy": None, "latency_ms": None,
                             "status": f"FAILED: {e}"})
            bar.progress((i + 1) / len(trips))
        df = pd.DataFrame(rows)
        st.session_state["all_df"] = df

    df = st.session_state.get("all_df")
    if df is not None:
        ok = df[df["status"] == "OK"]
        a, b, c = st.columns(3)
        a.metric("Trips succeeded", f"{len(ok)} / {len(df)}")
        a_mean = ok["accuracy"].mean() if len(ok) else None
        b.metric("Mean accuracy", f"{a_mean:.2%}" if a_mean is not None else "N/A")
        c.metric("Mean response time",
                 f"{ok['latency_ms'].mean():.0f} ms" if len(ok) else "N/A")
        st.dataframe(df)
        if len(ok):
            st.bar_chart(ok.set_index("trip")["accuracy"])

# ---------- History ----------
with tab_hist:
    if st.session_state["history"]:
        st.dataframe(pd.DataFrame(st.session_state["history"]))
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("No runs yet.")
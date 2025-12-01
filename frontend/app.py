import streamlit as st
import requests
import time
import math
import json
import networkx as nx
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# -------------------------
# Configuration
# -------------------------
st.set_page_config(
    page_title="TalentScope AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

UPLOAD_API_URL = "http://127.0.0.1:8000/upload-resume/"
GRAPH_API_URL = "http://127.0.0.1:8000/graph-data"
CANDIDATES_API_URL = "http://127.0.0.1:8000/candidates"

# -------------------------
# Page Header
# -------------------------
st.title("TalentScope AI - Resume Analyzer")
st.markdown("---")

# -------------------------
# Pipeline Phases
# -------------------------
pipeline_phases = [
    "Upload",
    "Cleaning",
    "Structured JSON Extraction",
    "Trait Inference",
    "FAISS Update",
    "Database + Graph Sync"
]

# -------------------------
# Section 1: Resume Upload
# -------------------------
st.header("1️⃣ Upload Resume & Pipeline Status")
st.info("Upload a candidate resume and watch the pipeline progress.")

uploaded_file = st.file_uploader(
    "Drag & drop a resume here or select a file",
    type=["pdf", "docx", "txt"]
)

if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = {phase: "Pending" for phase in pipeline_phases}
    st.session_state.upload_response = None

if uploaded_file:
    st.session_state.pipeline_status = {phase: "Pending" for phase in pipeline_phases}
    st.session_state.upload_response = None

    progress_bars = {}
    for phase in pipeline_phases:
        progress_bars[phase] = st.progress(0, text=f"{phase} - Pending")

    try:
        progress_bars["Upload"].progress(50, text="Upload - In progress")
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post(UPLOAD_API_URL, files=files)
        progress_bars["Upload"].progress(100, text="Upload - Completed")

        if response.status_code == 200:
            data = response.json()
            st.session_state.upload_response = data

            for phase in pipeline_phases[1:]:
                progress_bars[phase].progress(50, text=f"{phase} - In progress")
                time.sleep(0.5)
                progress_bars[phase].progress(100, text=f"{phase} - Completed")

            st.success("✅ Resume processed successfully!")

            with st.expander("View Structured JSON"):
                st.json(data["structured_output"])
            with st.expander("View Trait Scores"):
                st.json(data["traits_output"])
        else:
            st.error(f"❌ Upload failed: {response.text}")

    except Exception as e:
        st.error(f"❌ Error during upload: {e}")

st.markdown("---")

# -------------------------
# Section 2: Candidate Graph
# -------------------------
st.header("2️⃣ Candidate Graph")
st.info("Interactive candidate graph (click a candidate node to view details).")

@st.cache_data(ttl=300)
def fetch_graph_data():
    try:
        resp = requests.get(GRAPH_API_URL)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None

graph_data = fetch_graph_data()

selected_candidate_id = None

if graph_data:
    G = nx.Graph()
    node_map = {}
    for node in graph_data["nodes"]:
        node_id = node["id"]
        G.add_node(node_id, **node)
        node_map[node_id] = node

    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], label=edge["label"])

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text, node_color, node_ids = [], [], [], [], []
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        # Show only the label (candidate name or skill name)
        node_text.append(data.get('label', ''))
        node_color.append('#1DB954' if data.get('type') == 'Candidate' else '#FFD700')
        node_ids.append(n)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(color=node_color, size=20, line_width=2),
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        paper_bgcolor="#111111",
                        plot_bgcolor="#111111"
                    ))

    clicked_nodes = plotly_events(fig, click_event=True)

    if clicked_nodes:
        point_idx = clicked_nodes[0]['pointIndex']
        selected_candidate_id = node_ids[point_idx]

else:
    st.warning("No graph data available.")

st.markdown("---")

# -------------------------
# Section 3: Candidate Details
# -------------------------
st.header("3️⃣ Candidate Details")
st.info("Cards with structured JSON and trait scores will appear here.")

@st.cache_data(ttl=300)
def fetch_all_candidates():
    try:
        resp = requests.get(CANDIDATES_API_URL)
        if resp.status_code == 200:
            return resp.json()
        return []
    except:
        return []

candidate_data_list = fetch_all_candidates()

# Filter only selected candidate if a node is clicked
if selected_candidate_id:
    candidate_data_list = [c for c in candidate_data_list if str(c['candidate_id']) == str(selected_candidate_id)]

# Display candidate cards
cols_per_row = 3
num_rows = math.ceil(len(candidate_data_list) / cols_per_row)

for row in range(num_rows):
    cols = st.columns(cols_per_row, gap="medium")
    for i in range(cols_per_row):
        idx = row * cols_per_row + i
        if idx >= len(candidate_data_list):
            break
        data = candidate_data_list[idx]
        candidate_id = data['candidate_id']
        with cols[i]:
            st.markdown(f"""
            <div style="background-color:#222222;padding:15px;border-radius:10px;box-shadow:2px 2px 10px rgba(0,0,0,0.5);">
            <h3 style="color:#1DB954;">{data['structured_json'].get('name','Candidate')}</h3>
            <p style="color:#E0E0E0;">Email: {data['structured_json'].get('email','N/A')}</p>
            <p style="color:#E0E0E0;">Phone: {data['structured_json'].get('phone','N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

            traits = data.get("traits", {})
            if traits:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=list(traits.values()),
                    theta=list(traits.keys()),
                    fill='toself',
                    marker_color='#1DB954'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    paper_bgcolor='#222222',
                    font_color='#E0E0E0',
                    margin=dict(l=20,r=20,t=20,b=20)
                )
                st.plotly_chart(fig, use_container_width=True, key=f"radar_{candidate_id}")

            with st.expander("Structured JSON"):
                st.json(data["structured_json"])

            st.download_button(
                label="Download Resume JSON",
                data=json.dumps(data["structured_json"], indent=2),
                file_name=f"{candidate_id}_structured.json",
                mime="application/json"
            )
            st.download_button(
                label="Download Traits JSON",
                data=json.dumps(data["traits"], indent=2),
                file_name=f"{candidate_id}_traits.json",
                mime="application/json"
            )

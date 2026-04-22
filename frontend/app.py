import streamlit as st
import requests
import time
import math
import json
import networkx as nx
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import pandas as pd

# -------------------------
# Configuration
# -------------------------
st.set_page_config(
    page_title="TalentScope AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URLs
UPLOAD_API_URL          = "http://127.0.0.1:8000/upload-resume/"
GRAPH_API_URL           = "http://127.0.0.1:8000/graph-data"
CANDIDATES_API_URL      = "http://127.0.0.1:8000/candidates"
SEARCH_API_URL          = "http://127.0.0.1:8000/search"
JOBS_API_URL            = "http://127.0.0.1:8000/jobs"
SKILL_GAP_API_URL       = "http://127.0.0.1:8000/skill-gap"
RANK_CANDIDATES_API_URL = "http://127.0.0.1:8000/rank-candidates"
RECOMMEND_JOBS_API_URL  = "http://127.0.0.1:8000/recommend-jobs"
COMPARE_API_URL         = "http://127.0.0.1:8000/compare-candidates"
INTERVIEW_API_URL       = "http://127.0.0.1:8000/generate-interview"

# -------------------------
# Page Header
# -------------------------
st.title("TalentScope AI - Resume Analyzer")
st.markdown("---")

# -------------------------
# Pipeline Phases
# -------------------------
pipeline_phases = [
    "Upload", "Cleaning", "Structured JSON Extraction",
    "Trait Inference", "FAISS Update", "Database + Graph Sync"
]

# -------------------------
# Cached data fetchers
# -------------------------
@st.cache_data(ttl=300)
def fetch_all_candidates():
    try:
        resp = requests.get(CANDIDATES_API_URL)
        return resp.json() if resp.status_code == 200 else []
    except:
        return []

@st.cache_data(ttl=300)
def fetch_graph_data():
    try:
        resp = requests.get(GRAPH_API_URL)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=60)
def fetch_jobs():
    try:
        resp = requests.get(JOBS_API_URL)
        return resp.json() if resp.status_code == 200 else []
    except:
        return []

# -------------------------
# Section 1: Resume Upload
# -------------------------
st.header("1️⃣ Upload Resume & Pipeline Status")
st.info("Upload a candidate resume and watch the pipeline progress.")

uploaded_file = st.file_uploader(
    "Drag & drop a resume here or select a file",
    type=["pdf", "docx", "txt"]
)

if "upload_processed" not in st.session_state:
    st.session_state.upload_processed = False
    st.session_state.last_uploaded_filename = None
    st.session_state.upload_response = None

new_file_detected = (
    uploaded_file is not None
    and uploaded_file.name != st.session_state.last_uploaded_filename
)
if new_file_detected:
    st.session_state.upload_processed = False
    st.session_state.last_uploaded_filename = uploaded_file.name
    st.session_state.upload_response = None

if uploaded_file and not st.session_state.upload_processed:
    progress_bars = {phase: st.progress(0, text=f"{phase} - Pending") for phase in pipeline_phases}

    try:
        progress_bars["Upload"].progress(50, text="Upload - In progress")
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post(UPLOAD_API_URL, files=files)
        progress_bars["Upload"].progress(100, text="Upload - Completed")

        if response.status_code == 200:
            data = response.json()
            st.session_state.upload_response = data
            st.session_state.upload_processed = True

            for phase in pipeline_phases[1:]:
                progress_bars[phase].progress(50, text=f"{phase} - In progress")
                time.sleep(0.5)
                progress_bars[phase].progress(100, text=f"{phase} - Completed")

            fetch_all_candidates.clear()
            fetch_graph_data.clear()

            st.success("✅ Resume processed successfully!")
            with st.expander("View Structured JSON"):
                st.json(data.get("structured_output", {}))
            with st.expander("View Trait Scores"):
                st.json(data.get("traits_output", {}))
        else:
            st.error(f"❌ Upload failed: {response.text}")

    except Exception as e:
        st.error(f"❌ Error during upload: {e}")

elif uploaded_file and st.session_state.upload_processed and st.session_state.upload_response:
    st.success("✅ Resume already processed.")
    with st.expander("View Structured JSON"):
        st.json(st.session_state.upload_response.get("structured_output", {}))
    with st.expander("View Trait Scores"):
        st.json(st.session_state.upload_response.get("traits_output", {}))

st.markdown("---")

# -------------------------
# Section 2: Candidate Search
# -------------------------
st.header("🔍 Candidate Search")
st.info("Query the hybrid retriever to find the best candidate matches based on skills and experience.")

search_query = st.text_input("Enter search query (e.g., 'Machine Learning Engineer with Python')")
top_k = st.slider("Number of top candidates to retrieve", min_value=1, max_value=50, value=3)

if st.button("Search Candidates"):
    if search_query.strip():
        with st.spinner("Searching..."):
            try:
                search_payload = {"query": search_query, "top_k": top_k}
                response = requests.post(SEARCH_API_URL, json=search_payload)
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if results:
                        st.success(f"Found {len(results)} matching candidates!")
                        for i, res in enumerate(results):
                            with st.expander(f"Candidate Match {i+1} (Score: {res.get('composite_score', 0):.2f})"):
                                st.write(f"**Candidate ID:** {res.get('candidate_id')}")
                                st.write(f"**Snippet:** {res.get('snippet', '')[:150]}...")
                                if res.get("graph_path"):
                                    st.write(f"**Graph Connection:** {res['graph_path']}")
                                st.write(f"*(Vector Score: {res.get('dense_sim', 0):.2f}, Graph Score: {res.get('graph_score', 0):.2f})*")
                    else:
                        st.info("No matches found for your query.")
                else:
                    st.error(f"Search failed: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
    else:
        st.warning("Please enter a search query.")

st.markdown("---")

# -------------------------
# Section 3: Candidate Graph
# -------------------------
st.header("2️⃣ Candidate Graph")
st.info("Interactive candidate graph (click a candidate node to view details). Press 'Refresh Graph' after upload if new nodes don't appear.")

if st.button("🔄 Refresh Graph"):
    fetch_graph_data.clear()
    st.rerun()

graph_data = fetch_graph_data()
selected_candidate_id = None

if graph_data and graph_data.get("nodes"):
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
        node_text.append(data.get('label', ''))
        node_type = data.get('type', '')
        node_color.append('#1DB954' if node_type == 'Candidate' else '#FFD700')
        node_ids.append(n)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(color=node_color, size=20, line_width=2),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111"
        )
    )

    clicked_nodes = plotly_events(fig, click_event=True)
    if clicked_nodes:
        point_idx = clicked_nodes[0]['pointIndex']
        selected_candidate_id = node_ids[point_idx]
else:
    st.warning("No graph data available. Upload resumes to build the graph.")

st.markdown("---")

# -------------------------
# Section 4: Candidate Details
# -------------------------
st.header("3️⃣ Candidate Details")
st.info("Cards with structured JSON and trait scores. Press 'Refresh Candidates' if a newly uploaded resume isn't visible.")

if st.button("🔄 Refresh Candidates"):
    fetch_all_candidates.clear()
    st.rerun()

candidate_data_list = fetch_all_candidates()

unique_candidates = {}
for c in candidate_data_list:
    unique_candidates[str(c['candidate_id'])] = c
candidate_data_list = list(unique_candidates.values())

if selected_candidate_id:
    candidate_data_list = [c for c in candidate_data_list if str(c['candidate_id']) == str(selected_candidate_id)]

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
                    margin=dict(l=20, r=20, t=20, b=20)
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
                data=json.dumps(data.get("traits", {}), indent=2),
                file_name=f"{candidate_id}_traits.json",
                mime="application/json"
            )

st.markdown("---")

# =============================================================================
# Section 5: Skill Gap Analysis
# =============================================================================
st.header("4️⃣ Skill Gap Analysis")
st.info("Compare job requirements against candidate skills and rank candidates by fit.")

with st.expander("➕ Add a New Job Posting"):
    with st.form("add_job_form"):
        job_title_input  = st.text_input("Job Title", placeholder="e.g. Data Engineer")
        job_desc_input   = st.text_area("Job Description (optional)", placeholder="Brief description of the role...")
        job_skills_input = st.text_input(
            "Required Skills (comma-separated)",
            placeholder="e.g. Python, SQL, Docker, Spark"
        )
        submitted = st.form_submit_button("Add Job")

        if submitted:
            if not job_title_input.strip():
                st.warning("Job title is required.")
            elif not job_skills_input.strip():
                st.warning("At least one skill is required.")
            else:
                skills_list = [s.strip() for s in job_skills_input.split(",") if s.strip()]
                payload = {
                    "title": job_title_input.strip(),
                    "description": job_desc_input.strip(),
                    "skills": skills_list
                }
                try:
                    r = requests.post(JOBS_API_URL, json=payload)
                    if r.status_code == 201:
                        st.success(f"✅ Job '{job_title_input}' added (ID: {r.json()['job_id']})")
                        fetch_jobs.clear()
                    else:
                        st.error(f"Failed to add job: {r.text}")
                except Exception as ex:
                    st.error(f"Error: {ex}")

st.markdown("")

jobs = fetch_jobs()

if not jobs:
    st.warning("No jobs found. Add a job posting above to get started.")
else:
    job_options = {f"[{j['job_id']}] {j['title']}": j for j in jobs}
    selected_job_label = st.selectbox("🎯 Select a Job", list(job_options.keys()), key="sg_job_select")
    selected_job = job_options[selected_job_label]

    st.markdown(
        f"**Required Skills ({len(selected_job['skills'])}):** "
        + " · ".join(f"`{s}`" for s in selected_job["skills"])
    )
    st.markdown("")

    st.subheader("🔎 Individual Skill Gap")

    all_candidates_raw = fetch_all_candidates()
    unique_cands = {str(c['candidate_id']): c for c in all_candidates_raw}
    all_candidates_list = list(unique_cands.values())

    if not all_candidates_list:
        st.info("No candidates found. Upload resumes first.")
    else:
        cand_options = {
            f"[{c['candidate_id']}] {c['structured_json'].get('name', 'Unknown')}": c['candidate_id']
            for c in all_candidates_list
        }
        selected_cand_label = st.selectbox("👤 Select a Candidate", list(cand_options.keys()), key="sg_cand_select")
        selected_cand_id = cand_options[selected_cand_label]

        if st.button("Analyse Skill Gap"):
            with st.spinner("Analysing..."):
                try:
                    resp = requests.get(f"{SKILL_GAP_API_URL}/{selected_job['job_id']}/{selected_cand_id}")
                    if resp.status_code == 200:
                        gap = resp.json()
                        match_pct = gap["match_percentage"]
                        gauge_color = "#1DB954" if match_pct >= 70 else "#FFA500" if match_pct >= 40 else "#E74C3C"

                        st.markdown(
                            f"""
                            <div style="background:#1a1a2e;border-radius:12px;padding:20px;margin-bottom:16px;text-align:center;">
                                <h2 style="color:{gauge_color};margin:0;">{match_pct}%</h2>
                                <p style="color:#aaa;margin:4px 0 0 0;">Skill Match for
                                <strong style="color:#fff">{gap['candidate_name']}</strong>
                                → <strong style="color:#fff">{gap['job_title']}</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        col_match, col_miss = st.columns(2)
                        with col_match:
                            st.markdown(f"<h4 style='color:#1DB954;'>✅ Matched Skills ({gap['matched_count']})</h4>", unsafe_allow_html=True)
                            for skill in gap["matched_skills"]:
                                st.markdown(f"<span style='background:#1DB95422;color:#1DB954;padding:4px 10px;border-radius:20px;margin:3px;display:inline-block;'>{skill}</span>", unsafe_allow_html=True)
                        with col_miss:
                            st.markdown(f"<h4 style='color:#E74C3C;'>❌ Missing Skills ({gap['missing_count']})</h4>", unsafe_allow_html=True)
                            for skill in gap["missing_skills"]:
                                st.markdown(f"<span style='background:#E74C3C22;color:#E74C3C;padding:4px 10px;border-radius:20px;margin:3px;display:inline-block;'>{skill}</span>", unsafe_allow_html=True)
                    else:
                        st.error(f"Error: {resp.json().get('detail', resp.text)}")
                except Exception as ex:
                    st.error(f"Error: {ex}")

    st.markdown("")

    st.subheader("🏆 Rank Candidates by Skill Match")

    if st.button("Rank All Candidates"):
        with st.spinner("Ranking candidates..."):
            try:
                resp = requests.get(f"{RANK_CANDIDATES_API_URL}/{selected_job['job_id']}")
                if resp.status_code == 200:
                    ranking_data = resp.json()
                    candidates_ranked = ranking_data["candidates"]
                    if not candidates_ranked:
                        st.info("No candidates to rank.")
                    else:
                        st.success(f"Ranked **{len(candidates_ranked)}** candidates for **{ranking_data['job_title']}**")
                        for cand in candidates_ranked:
                            match_pct = cand["match_percentage"]
                            bar_color = "#1DB954" if match_pct >= 70 else "#FFA500" if match_pct >= 40 else "#E74C3C"
                            medal = "🥇" if cand["rank"] == 1 else "🥈" if cand["rank"] == 2 else "🥉" if cand["rank"] == 3 else f"#{cand['rank']}"
                            missing_preview = f"&nbsp;|&nbsp; Missing: <em>{', '.join(cand['missing_skills'][:5])}</em>" if cand["missing_skills"] else ""

                            st.markdown(
                                f"""
                                <div style="background:#1a1a2e;border-radius:10px;padding:14px 18px;margin-bottom:10px;border-left:5px solid {bar_color};">
                                    <div style="display:flex;justify-content:space-between;align-items:center;">
                                        <span style="font-size:1.1em;color:#fff;">
                                            {medal} &nbsp;<strong>{cand['candidate_name']}</strong>
                                            <span style="color:#888;font-size:0.85em;"> · ID {cand['candidate_id']}</span>
                                        </span>
                                        <span style="font-size:1.4em;font-weight:bold;color:{bar_color};">{match_pct}%</span>
                                    </div>
                                    <div style="background:#333;border-radius:6px;height:8px;margin-top:8px;">
                                        <div style="background:{bar_color};width:{match_pct}%;height:8px;border-radius:6px;"></div>
                                    </div>
                                    <div style="margin-top:8px;font-size:0.85em;color:#aaa;">
                                        ✅ {cand['matched_count']} matched &nbsp;|&nbsp; ❌ {cand['missing_count']} missing{missing_preview}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except Exception as ex:
                st.error(f"Error: {ex}")

st.markdown("---")

# =============================================================================
# Section 6: Job Recommendations for a Candidate
# =============================================================================
st.header("5️⃣ Job Recommendations for a Candidate")
st.info("Select a candidate to see which jobs best match their skills, ranked by compatibility.")

all_candidates_for_rec = fetch_all_candidates()
unique_cands_rec = {str(c['candidate_id']): c for c in all_candidates_for_rec}
candidates_for_rec = list(unique_cands_rec.values())

if not candidates_for_rec:
    st.warning("No candidates found. Upload resumes first.")
else:
    rec_cand_options = {
        f"[{c['candidate_id']}] {c['structured_json'].get('name', 'Unknown')}": c['candidate_id']
        for c in candidates_for_rec
    }
    selected_rec_cand_label = st.selectbox(
        "👤 Select a Candidate",
        list(rec_cand_options.keys()),
        key="rec_cand_select"
    )
    selected_rec_cand_id = rec_cand_options[selected_rec_cand_label]

    if st.button("Get Job Recommendations"):
        with st.spinner("Finding best-matching jobs..."):
            try:
                resp = requests.get(f"{RECOMMEND_JOBS_API_URL}/{selected_rec_cand_id}")
                if resp.status_code == 200:
                    rec_data = resp.json()
                    candidate_name = rec_data["candidate_name"]
                    candidate_skills = rec_data["candidate_skills"]
                    recommended_jobs = rec_data["recommended_jobs"]

                    st.markdown(
                        f"**{candidate_name}'s Skills ({len(candidate_skills)}):** "
                        + " · ".join(f"`{s}`" for s in candidate_skills)
                    )
                    st.markdown("")

                    if not recommended_jobs:
                        st.info("No jobs available to match against. Add job postings first.")
                    else:
                        st.success(f"Found **{len(recommended_jobs)}** jobs ranked by compatibility for **{candidate_name}**")

                        for job in recommended_jobs:
                            match_pct = job["match_percentage"]
                            bar_color = "#1DB954" if match_pct >= 70 else "#FFA500" if match_pct >= 40 else "#E74C3C"
                            medal = "🥇" if job["rank"] == 1 else "🥈" if job["rank"] == 2 else "🥉" if job["rank"] == 3 else f"#{job['rank']}"

                            st.markdown(
                                f"""
                                <div style="background:#1a1a2e;border-radius:10px;padding:16px 20px;
                                            margin-bottom:12px;border-left:5px solid {bar_color};">
                                    <div style="display:flex;justify-content:space-between;align-items:center;">
                                        <span style="font-size:1.15em;color:#fff;">
                                            {medal} &nbsp;<strong>{job['job_title']}</strong>
                                            <span style="color:#888;font-size:0.82em;"> · Job ID {job['job_id']}</span>
                                        </span>
                                        <span style="font-size:1.5em;font-weight:bold;color:{bar_color};">{match_pct}%</span>
                                    </div>
                                    <div style="background:#333;border-radius:6px;height:8px;margin-top:10px;">
                                        <div style="background:{bar_color};width:{match_pct}%;height:8px;border-radius:6px;"></div>
                                    </div>
                                    <div style="margin-top:10px;font-size:0.85em;color:#aaa;">
                                        ✅ {job['matched_count']}/{job['total_required_skills']} skills matched
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except Exception as ex:
                st.error(f"Error: {ex}")

st.markdown("---")

# =============================================================================
# Section 7: Side-by-Side Candidate Comparison
# =============================================================================
st.header("6️⃣ Side-by-Side Candidate Comparison")
st.info("Select 2–4 candidates to compare using structured data, traits, embedding similarity, and graph signals.")

candidate_data_list = fetch_all_candidates()
unique_candidates = {str(c['candidate_id']): c for c in candidate_data_list}
candidate_data_list = list(unique_candidates.values())

if not candidate_data_list:
    st.warning("No candidates available. Upload resumes first.")
else:
    cand_options = {
        f"[{c['candidate_id']}] {c['structured_json'].get('name', 'Unknown Candidate')}": c
        for c in candidate_data_list
    }

    selected_labels = st.multiselect(
        "👥 Select candidates to compare (maximum 4)",
        options=list(cand_options.keys()),
        max_selections=4,
        key="side_by_side_select"
    )

    if len(selected_labels) >= 2 and st.button("🔍 Generate Side-by-Side Comparison", type="primary"):
        selected_cands = [cand_options[label] for label in selected_labels]
        n = len(selected_cands)

        st.success(f"Comparing {n} candidates side-by-side")

        cols = st.columns(n, gap="medium")

        for i, cand in enumerate(selected_cands):
            with cols[i]:
                name = cand['structured_json'].get('name', 'Unknown')
                cid = cand['candidate_id']
                st.markdown(f"""
                    <div style="background:#1a1a2e; padding:14px; border-radius:10px; text-align:center; border-bottom:4px solid #1DB954;">
                        <h3 style="margin:0; color:#1DB954;">{name}</h3>
                        <small style="color:#888;">ID: {cid}</small>
                    </div>
                """, unsafe_allow_html=True)

                traits = cand.get("traits", {})
                if traits:
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=list(traits.values()),
                        theta=[t.replace('_', ' ').title() for t in traits.keys()],
                        fill='toself',
                        marker_color='#1DB954'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                        height=320,
                        paper_bgcolor='#222222',
                        font_color='#E0E0E0'
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"cmp_radar_{cid}")

        # Trait Heatmap
        st.subheader("Trait Heatmap Comparison")
        trait_names = list(selected_cands[0].get("traits", {}).keys())
        if trait_names:
            heatmap_data = [[cand.get("traits", {}).get(trait, 0.0) for cand in selected_cands] for trait in trait_names]

            fig_heat = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=[cand['structured_json'].get('name', f"ID{cand['candidate_id']}")[:18] for cand in selected_cands],
                y=[t.replace('_', ' ').title() for t in trait_names],
                colorscale='RdYlGn',
                text=[[f"{val:.2f}" for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                hoverongaps=False
            ))
            fig_heat.update_layout(height=420, paper_bgcolor='#111111', font_color='#E0E0E0')
            st.plotly_chart(fig_heat, use_container_width=True)

        # Detailed Comparison Table
        st.subheader("Detailed Comparison Table")
        compare_rows = []
        for trait in trait_names:
            row = {"Metric": trait.replace('_', ' ').title()}
            for cand in selected_cands:
                row[cand['structured_json'].get('name', f"ID{cand['candidate_id']}")] = cand.get("traits", {}).get(trait, 0.0)
            compare_rows.append(row)

        for field in ["email", "phone"]:
            row = {"Metric": field.title()}
            for cand in selected_cands:
                row[cand['structured_json'].get('name', f"ID{cand['candidate_id']}")] = cand['structured_json'].get(field, "N/A")
            compare_rows.append(row)

        df_compare = pd.DataFrame(compare_rows)

        def highlight_traits(val):
            if isinstance(val, float):
                if val >= 0.75: return 'background-color: #1DB954; color: white'
                elif val >= 0.5: return 'background-color: #FFA500; color: white'
                else: return 'background-color: #E74C3C; color: white'
            return ''

        styled_df = (
            df_compare.style
            .applymap(highlight_traits, subset=df_compare.columns[1:])
            .format({col: "{:.2f}" for col in df_compare.columns if col != "Metric"})
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.markdown("---")

# =============================================================================
# Section 8: Automated Custom Interview Generator
# =============================================================================
st.header("7️⃣ Automated Custom Interview Generator")
st.info("Generate role- and candidate-specific interview questions using skill gaps and inferred traits.")

candidate_data_list = fetch_all_candidates()
unique_candidates = {str(c['candidate_id']): c for c in candidate_data_list}
candidate_data_list = list(unique_candidates.values())

jobs = fetch_jobs()

if not candidate_data_list or not jobs:
    st.warning("Please upload resumes and add jobs to use this feature.")
else:
    cand_options = {
        f"[{c['candidate_id']}] {c['structured_json'].get('name', 'Unknown')}": c['candidate_id']
        for c in candidate_data_list
    }
    selected_cand_label = st.selectbox(
        "👤 Select Candidate",
        list(cand_options.keys()),
        key="interview_cand_select"
    )
    selected_candidate_id = cand_options[selected_cand_label]

    job_options = {f"[{j['job_id']}] {j['title']}": j for j in jobs}
    selected_job_label = st.selectbox(
        "🎯 Select Job",
        list(job_options.keys()),
        key="interview_job_select"
    )
    selected_job = job_options[selected_job_label]

    col1, col2 = st.columns([3, 1])
    with col1:
        num_questions = st.slider("Number of Questions", min_value=4, max_value=12, value=8)
    with col2:
        focus = st.selectbox(
            "Focus Area",
            ["balanced", "technical", "behavioral", "leadership"],
            index=0
        )

    if st.button("🎯 Generate Custom Interview Kit", type="primary"):
        with st.spinner("Generating personalized interview questions..."):
            try:
                payload = {
                    "candidate_id": selected_candidate_id,
                    "job_id": selected_job['job_id'],
                    "num_questions": num_questions,
                    "focus": focus
                }
                resp = requests.post(INTERVIEW_API_URL, json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    questions = data.get("questions", [])

                    st.success(
                        f"✅ Interview Kit generated for **{data['candidate_name']}** "
                        f"applying for **{data['job_title']}**"
                    )

                    st.markdown(
                        f"**Summary:** {data.get('summary', 'Custom interview kit generated based on skill gaps and traits.')}"
                    )
                    st.markdown(f"**Total Questions:** {len(questions)}")
                    st.markdown("---")

                    difficulty_color = {
                        "Easy":   "#1DB954",
                        "Medium": "#FFA500",
                        "Hard":   "#E74C3C"
                    }

                    for i, q in enumerate(questions, 1):
                        question_text = q.get("question", "No question text provided.")
                        difficulty    = q.get("difficulty", "Medium")
                        category      = q.get("category", "General")
                        rubric        = q.get("rubric", "No rubric provided.")
                        evidence      = q.get("evidence", [])

                        d_color = difficulty_color.get(difficulty, "#FFA500")

                        # Expander label shows number + short preview only
                        preview = question_text[:60] + "..." if len(question_text) > 60 else question_text
                        with st.expander(f"Q{i}: {preview}", expanded=(i <= 2)):

                            # ── Full question text ──────────────────────────
                            st.markdown(
                                f"""
                                <div style="background:#1a1a2e;border-radius:10px;padding:14px 18px;
                                            margin-bottom:12px;border-left:5px solid {d_color};">
                                    <p style="color:#ffffff;font-size:1.05em;margin:0;line-height:1.6;">
                                        {question_text}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            # ── Metadata badges ─────────────────────────────
                            st.markdown(
                                f"**Difficulty:** "
                                f"<span style='background:{d_color}22;color:{d_color};"
                                f"padding:3px 10px;border-radius:12px;font-size:0.85em;'>{difficulty}</span>"
                                f"&nbsp;&nbsp;**Category:** "
                                f"<span style='background:#ffffff22;color:#ffffff;"
                                f"padding:3px 10px;border-radius:12px;font-size:0.85em;'>{category}</span>",
                                unsafe_allow_html=True
                            )

                            st.markdown("")

                            # ── Rubric ──────────────────────────────────────
                            st.markdown("**📋 Evaluation Rubric:**")
                            st.info(rubric)

                            # ── Evidence ────────────────────────────────────
                            if evidence:
                                st.markdown("**🔍 Why This Question Was Selected:**")
                                for ev in evidence:
                                    st.markdown(
                                        f"<span style='background:#ffffff11;color:#aaaaaa;"
                                        f"padding:3px 10px;border-radius:8px;margin:3px;"
                                        f"display:inline-block;font-size:0.85em;'>• {ev}</span>",
                                        unsafe_allow_html=True
                                    )

                    # ── Download Kit ────────────────────────────────────────
                    st.markdown("---")
                    kit_json = json.dumps(data, indent=2)
                    st.download_button(
                        label="⬇️ Download Full Interview Kit (JSON)",
                        data=kit_json,
                        file_name=f"interview_kit_{data['candidate_name'].replace(' ', '_')}_{data['job_title'].replace(' ', '_')}.json",
                        mime="application/json"
                    )

                else:
                    err_detail = ""
                    try:
                        err_detail = resp.json().get("detail", resp.text)
                    except Exception:
                        err_detail = resp.text
                    st.error(f"Failed to generate interview: {err_detail}")

            except Exception as e:
                st.error(f"Error generating interview: {e}")

st.markdown("---")
# =============================================================================
# Section 9: Career Trajectory Modeling (NEW)
# =============================================================================
st.header("8️⃣ Career Trajectory Modeling")
st.info("Predict next roles, time-to-promotion, and probabilistic career paths using Graph-RAG.")

candidate_data_list = fetch_all_candidates()
unique_candidates = {str(c['candidate_id']): c for c in candidate_data_list}
candidate_data_list = list(unique_candidates.values())

if not candidate_data_list:
    st.warning("No candidates available. Upload resumes first.")
else:
    cand_options = {
        f"[{c['candidate_id']}] {c['structured_json'].get('name', 'Unknown')}": c['candidate_id']
        for c in candidate_data_list
    }

    selected_cand_label = st.selectbox("👤 Select Candidate", list(cand_options.keys()), key="trajectory_cand")
    selected_candidate_id = cand_options[selected_cand_label]

    num_paths = st.slider("Number of Predicted Career Paths", 2, 5, 3)

    if st.button("🔮 Predict Career Trajectory", type="primary"):
        with st.spinner("Analyzing career graph and predicting future paths..."):
            try:
                payload = {
                    "candidate_id": selected_candidate_id,
                    "num_paths": num_paths
                }
                resp = requests.post("http://127.0.0.1:8000/career-trajectory", json=payload)

                if resp.status_code == 200:
                    data = resp.json()

                    st.success(f"Career Trajectory for **{data['candidate_name']}**")
                    st.markdown(f"**Current Role:** {data['current_role']} at {data['current_company']}")
                    st.markdown(f"**Summary:** {data['summary']}")

                    for i, path in enumerate(data["predicted_paths"], 1):
                        with st.expander(f"Path {i} — {path['predicted_role']} ({path['probability']}% probability)", expanded=True):
                            st.write(f"**Company Type:** {path['company_type']}")
                            st.write(f"**Estimated Time to Promotion:** {path['time_to_promotion_years']} years")
                            st.write(f"**Key Skills Needed:** {', '.join(path['key_skills_needed'])}")
                            st.markdown("**Rationale:**")
                            st.info(path['rationale'])
                            if path['evidence']:
                                st.markdown("**Evidence from Profile:**")
                                for ev in path['evidence']:
                                    st.caption(f"• {ev}")
                else:
                    st.error(f"Failed: {resp.text}")

            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
# =============================================================================
# Section 10: Team Fit Analysis
# =============================================================================
TEAM_FIT_API_URL = "http://127.0.0.1:8000/team-fit"

st.header("9️⃣ Team Fit Analysis")
st.info("Evaluate how well a candidate would integrate into an existing team using trait complementarity, skill proximity, and graph-based signals.")

candidate_data_list = fetch_all_candidates()
unique_candidates   = {str(c['candidate_id']): c for c in candidate_data_list}
candidate_data_list = list(unique_candidates.values())

if not candidate_data_list:
    st.warning("No candidates available. Upload resumes first.")
else:
    all_cand_options = {
        f"[{c['candidate_id']}] {c['structured_json'].get('name', 'Unknown')}": c['candidate_id']
        for c in candidate_data_list
    }

    # ── Candidate being evaluated ────────────────────────────────────────────
    tf_cand_label = st.selectbox(
        "👤 Candidate to Evaluate",
        list(all_cand_options.keys()),
        key="tf_candidate_select"
    )
    tf_candidate_id = all_cand_options[tf_cand_label]

    # ── Existing team members (exclude selected candidate) ───────────────────
    team_options = {
        label: cid
        for label, cid in all_cand_options.items()
        if cid != tf_candidate_id
    }

    selected_team_labels = st.multiselect(
        "👥 Select Existing Team Members",
        options=list(team_options.keys()),
        key="tf_team_select"
    )
    selected_team_ids = [team_options[l] for l in selected_team_labels]

    if len(selected_team_ids) == 0:
        st.info("Select at least one team member to run the analysis.")

    elif st.button("🤝 Analyse Team Fit", type="primary"):
        with st.spinner("Analysing trait complementarity and skill proximity..."):
            try:
                payload = {
                    "candidate_id": tf_candidate_id,
                    "team_member_ids": selected_team_ids
                }
                resp = requests.post(TEAM_FIT_API_URL, json=payload)

                if resp.status_code == 200:
                    data = resp.json()

                    score      = data["overall_team_fit_score"]
                    score_pct  = round(score * 100)
                    score_color = (
                        "#1DB954" if score >= 0.70 else
                        "#FFA500" if score >= 0.45 else
                        "#E74C3C"
                    )

                    # ── Overall score card ───────────────────────────────────
                    st.markdown(
                        f"""
                        <div style="background:#1a1a2e;border-radius:14px;padding:24px;
                                    margin-bottom:20px;text-align:center;
                                    border:1px solid {score_color}44;">
                            <h1 style="color:{score_color};margin:0;font-size:3em;">{score_pct}%</h1>
                            <p style="color:#aaa;margin:6px 0 0 0;font-size:1.1em;">
                                Overall Team Fit —
                                <strong style="color:#fff">{data['candidate_name']}</strong>
                                joining a team of
                                <strong style="color:#fff">{data['team_size']}</strong>
                            </p>
                            <p style="color:#888;margin:8px 0 0 0;font-size:0.9em;font-style:italic;">
                                {data['summary']}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # ── Score bar breakdown ──────────────────────────────────
                    st.markdown(
                        f"""
                        <div style="background:#111;border-radius:8px;height:10px;margin-bottom:20px;">
                            <div style="background:{score_color};width:{score_pct}%;
                                        height:10px;border-radius:8px;"></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # ── Trait radar: candidate vs team average ───────────────
                    st.subheader("🧠 Trait Profile: Candidate vs Team Average")
                    trait_labels = [t.replace("_", " ").title() for t in data["candidate_traits"].keys()]
                    cand_vals    = list(data["candidate_traits"].values())
                    team_vals    = list(data["team_avg_traits"].values())

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=cand_vals,
                        theta=trait_labels,
                        fill="toself",
                        name=data["candidate_name"],
                        line=dict(color="#1DB954"),
                        fillcolor="rgba(29,185,84,0.15)"
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=team_vals,
                        theta=trait_labels,
                        fill="toself",
                        name="Team Average",
                        line=dict(color="#FFD700", dash="dot"),
                        fillcolor="rgba(255,215,0,0.10)"
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True,
                        height=420,
                        paper_bgcolor="#111111",
                        font_color="#E0E0E0",
                        legend=dict(bgcolor="#1a1a2e", bordercolor="#333", borderwidth=1)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, key="tf_radar")

                    # ── Skills breakdown ─────────────────────────────────────
                    st.subheader("🛠️ Skill Contribution")
                    col_overlap, col_unique = st.columns(2)

                    with col_overlap:
                        st.markdown(
                            f"<h4 style='color:#FFA500;'>🔗 Shared with Team ({len(data['skill_overlap'])})</h4>",
                            unsafe_allow_html=True
                        )
                        if data["skill_overlap"]:
                            pills = " ".join(
                                f"<span style='background:#FFA50022;color:#FFA500;"
                                f"padding:4px 10px;border-radius:20px;margin:3px;"
                                f"display:inline-block;font-size:0.85em;'>{s}</span>"
                                for s in data["skill_overlap"]
                            )
                            st.markdown(pills, unsafe_allow_html=True)
                        else:
                            st.caption("No overlapping skills found.")

                    with col_unique:
                        st.markdown(
                            f"<h4 style='color:#1DB954;'>✨ Unique Contributions ({len(data['unique_skills_brought'])})</h4>",
                            unsafe_allow_html=True
                        )
                        if data["unique_skills_brought"]:
                            pills = " ".join(
                                f"<span style='background:#1DB95422;color:#1DB954;"
                                f"padding:4px 10px;border-radius:20px;margin:3px;"
                                f"display:inline-block;font-size:0.85em;'>{s}</span>"
                                for s in data["unique_skills_brought"]
                            )
                            st.markdown(pills, unsafe_allow_html=True)
                        else:
                            st.caption("No unique skills — candidate overlaps entirely with the team.")

                    # ── Pairwise member scores ───────────────────────────────
                    st.subheader("🔍 Pairwise Fit with Each Team Member")
                    for member in data["member_scores"]:
                        fit     = member["overall_pairwise_fit"]
                        fit_pct = round(fit * 100)
                        m_color = (
                            "#1DB954" if fit >= 0.70 else
                            "#FFA500" if fit >= 0.45 else
                            "#E74C3C"
                        )
                        st.markdown(
                            f"""
                            <div style="background:#1a1a2e;border-radius:10px;padding:14px 18px;
                                        margin-bottom:10px;border-left:5px solid {m_color};">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <span style="color:#fff;font-size:1.05em;">
                                        <strong>{member['member_name']}</strong>
                                        <span style="color:#888;font-size:0.82em;"> · ID {member['member_id']}</span>
                                    </span>
                                    <span style="font-size:1.4em;font-weight:bold;color:{m_color};">{fit_pct}%</span>
                                </div>
                                <div style="background:#333;border-radius:6px;height:6px;margin-top:10px;">
                                    <div style="background:{m_color};width:{fit_pct}%;height:6px;border-radius:6px;"></div>
                                </div>
                                <div style="margin-top:8px;font-size:0.82em;color:#aaa;">
                                    Trait Complementarity: <strong style="color:#fff">{round(member['trait_complementarity_score']*100)}%</strong>
                                    &nbsp;|&nbsp;
                                    Skill Proximity: <strong style="color:#fff">{round(member['skill_proximity_score']*100)}%</strong>
                                    &nbsp;|&nbsp;
                                    Graph Shared Skills: <strong style="color:#fff">{member['graph_shared_skills']}</strong>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # ── Counterfactual suggestions ───────────────────────────
                    if data.get("counterfactual_suggestions"):
                        st.subheader("💡 Counterfactual Pairing Suggestions")
                        for suggestion in data["counterfactual_suggestions"]:
                            st.markdown(
                                f"""
                                <div style="background:#1a1a2e;border-radius:8px;padding:12px 16px;
                                            margin-bottom:8px;border-left:4px solid #7B61FF;">
                                    <p style="color:#E0E0E0;margin:0;font-size:0.95em;">💬 {suggestion}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    # ── LLM analysis ─────────────────────────────────────────
                    with st.expander("📄 Full LLM Team Fit Analysis", expanded=False):
                        st.markdown(
                            f"<p style='color:#E0E0E0;line-height:1.8;'>{data['llm_analysis']}</p>",
                            unsafe_allow_html=True
                        )

                    # ── Download ─────────────────────────────────────────────
                    st.markdown("---")
                    st.download_button(
                        label="⬇️ Download Team Fit Report (JSON)",
                        data=json.dumps(data, indent=2),
                        file_name=f"team_fit_{data['candidate_name'].replace(' ', '_')}.json",
                        mime="application/json"
                    )

                else:
                    try:
                        err = resp.json().get("detail", resp.text)
                    except Exception:
                        err = resp.text
                    st.error(f"Team fit analysis failed: {err}")

            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

st.markdown("---")
st.caption("TalentScope AI - Graph-RAG Powered Talent Intelligence")

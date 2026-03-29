import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import requests
import time
import os
from datetime import datetime

import bcrypt
from huggingface_hub import hf_hub_download
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx, degree


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GraphSentry",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

#MainMenu, footer {visibility: hidden;}
div[data-testid="stDecoration"] {display: none;}

/* Sidebar collapse/expand button icon */
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="collapsedControl"] button span {
    font-family: 'Material Symbols Rounded' !important;
    font-size: 1.25rem !important;
    color: #71717a !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {
    background: #0a0a0b;
    border-right: 1px solid #27272a;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* --- Metrics --- */
div[data-testid="stMetric"] {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 0.5rem;
    padding: 1rem 1.25rem;
}
div[data-testid="stMetric"] label {
    color: #a1a1aa; font-size: 0.75rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.75rem; font-weight: 700; color: #fafafa;
}

/* --- Data elements --- */
div[data-testid="stDataFrame"] {
    border: 1px solid #27272a; border-radius: 0.5rem; overflow: hidden;
}
div.stPlotlyChart {
    border: 1px solid #27272a; border-radius: 0.5rem; overflow: hidden;
}

/* --- Inputs --- */
div[data-baseweb="select"] > div {
    border-color: #27272a; border-radius: 0.375rem;
    background: #18181b; color: #fafafa;
}
div[data-baseweb="input"] > div {
    border-color: #27272a; border-radius: 0.375rem;
    background: #18181b; color: #fafafa;
}
input[data-baseweb="input"] {
    background: #18181b; color: #fafafa;
}
input::placeholder { color: #52525b; }

/* --- Multiselect tags --- */
span[data-baseweb="tag"] {
    background: #27272a !important;
    color: #fafafa !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 0.25rem !important;
}
span[data-baseweb="tag"] span { color: #fafafa !important; }
span[data-baseweb="tag"] svg { fill: #a1a1aa !important; }
div[data-baseweb="select"] svg { color: #71717a; }
div[data-baseweb="popover"] ul {
    background: #18181b; border: 1px solid #27272a;
}
div[data-baseweb="popover"] li {
    background: #18181b; color: #fafafa;
}
div[data-baseweb="popover"] li:hover {
    background: #27272a;
}

/* --- Buttons --- */
button[kind="primary"],
button[data-testid="stBaseButton-primary"],
div[data-testid="stFormSubmitButton"] button,
button[type="submit"] {
    background: #fafafa !important; color: #09090b !important;
    border: none !important; border-radius: 0.375rem;
    font-weight: 600 !important; font-size: 0.875rem !important;
    transition: background 150ms;
}
button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #d4d4d8 !important; color: #09090b !important;
}
div[data-testid="stFormSubmitButton"] button p {
    color: #09090b !important; font-weight: 600 !important;
}

/* --- Forms --- */
div[data-testid="stForm"] {
    border-color: #27272a;
    border-radius: 0.5rem;
    padding: 1.75rem 1.75rem 1.5rem;
    background: #0a0a0b;
}
div[data-testid="stForm"] label {
    font-size: 0.85rem; color: #a1a1aa; font-weight: 500;
}
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
    background: transparent; color: #fafafa;
    border: 1px solid #27272a; border-radius: 0.375rem;
    font-weight: 500; font-size: 0.875rem;
    transition: background 150ms;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
    background: #18181b;
}

/* --- Tabs --- */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
    color: #71717a;
    padding: 0.625rem 1rem;
    background: transparent;
    border: none;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    transition: color 150ms, border-color 150ms;
}
button[data-baseweb="tab"]:hover { color: #a1a1aa; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #fafafa;
    border-bottom: 2px solid #fafafa;
}
div[data-baseweb="tab-list"] {
    border-bottom: 1px solid #27272a;
    gap: 0;
}

/* --- Custom classes --- */
.section-header {
    font-size: 0.7rem; font-weight: 600; color: #71717a;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 0.75rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #27272a;
}
.stat-row {
    display: flex; justify-content: space-between;
    padding: 0.5rem 0; border-bottom: 1px solid #27272a; font-size: 0.875rem;
}
.stat-label { color: #71717a; }
.stat-value { color: #fafafa; font-weight: 600; }

.verdict-high {
    background: rgba(239, 68, 68, 0.15); color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    padding: 0.75rem 1rem; border-radius: 0.5rem;
    font-weight: 600; text-align: center; font-size: 1rem; margin-bottom: 1rem;
}
.verdict-medium {
    background: rgba(234, 179, 8, 0.15); color: #facc15;
    border: 1px solid rgba(234, 179, 8, 0.4);
    padding: 0.75rem 1rem; border-radius: 0.5rem;
    font-weight: 600; text-align: center; font-size: 1rem; margin-bottom: 1rem;
}
.verdict-low {
    background: rgba(34, 197, 94, 0.15); color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.4);
    padding: 0.75rem 1rem; border-radius: 0.5rem;
    font-weight: 600; text-align: center; font-size: 1rem; margin-bottom: 1rem;
}

.match-card {
    background: #18181b; border: 1px solid #27272a;
    border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.75rem;
    transition: border-color 150ms;
}
.match-card:hover { border-color: #3f3f46; }

.page-header {
    font-size: 1.5rem; font-weight: 700; color: #fafafa;
    margin-bottom: 0.25rem;
}
.page-desc {
    font-size: 0.85rem; color: #71717a; margin-bottom: 1.5rem;
}

.sidebar-logo {
    font-size: 1.125rem; font-weight: 700; color: #fafafa;
    display: flex; align-items: center; gap: 0.5rem;
    padding-bottom: 1.5rem; margin-bottom: 1rem;
    border-bottom: 1px solid #27272a;
}
.sidebar-section {
    font-size: 0.65rem; font-weight: 600; color: #52525b;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 1.5rem; margin-bottom: 0.5rem;
}
.sidebar-footer {
    position: fixed; bottom: 1rem; font-size: 0.7rem; color: #3f3f46;
}

.history-row {
    background: #18181b; border: 1px solid #27272a;
    border-radius: 0.5rem; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    display: flex; justify-content: space-between; align-items: center;
}
.history-addr {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.8rem; color: #fafafa;
}
.history-meta { font-size: 0.75rem; color: #71717a; }

.empty-state {
    text-align: center; padding: 3rem 1rem; color: #52525b;
}
.empty-state-title { font-size: 1rem; font-weight: 600; color: #71717a; margin-bottom: 0.25rem; }
.empty-state-desc { font-size: 0.85rem; }

/* --- Hide radio keyboard tooltip --- */
div[data-testid="InputInstructions"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'investigation_history' not in st.session_state:
    st.session_state.investigation_history = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Dashboard'


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------



def authenticate(email, password):
    """Validate credentials against secrets.toml using bcrypt."""
    try:
        creds = st.secrets.get("credentials", {})
    except Exception as e:
        st.error(f"DEBUG: secrets error: {e}")
        return False, None

    st.info(f"DEBUG: creds keys = {list(creds.keys()) if hasattr(creds, 'keys') else type(creds)}")

    for user_key, user_data in creds.items():
        if user_data.get("email", "").lower() == email.lower():
            stored_hash = user_data.get("password", "")
            st.info(f"DEBUG: matched email, hash starts with: {stored_hash[:20]}")
            try:
                if bcrypt.checkpw(password.encode(), stored_hash.encode()):
                    return True, {
                        'name': user_data.get('name', email),
                        'email': user_data.get('email', email),
                        'role': user_data.get('role', 'analyst'),
                    }
                else:
                    st.error("DEBUG: bcrypt check returned False")
            except Exception as e:
                st.error(f"DEBUG: bcrypt error: {e}")
    return False, None


def show_login():
    # Hide sidebar on login page
    st.markdown('<style>section[data-testid="stSidebar"]{display:none;}</style>',
                unsafe_allow_html=True)

    st.markdown('<div style="height:15vh;"></div>', unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.4, 1])
    with col_m:
        st.markdown(
            '<div style="text-align:center;font-size:1.75rem;font-weight:700;color:#fafafa;'
            'margin-bottom:0.35rem;letter-spacing:-0.02em;">GraphSentry</div>'
            '<div style="text-align:center;font-size:0.85rem;color:#52525b;margin-bottom:2.5rem;">'
            'Sign in to the forensic analyst dashboard</div>',
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            email = st.text_input("Email", placeholder="analyst@graphsentry.io")
            st.markdown('<div style="height:0.15rem;"></div>', unsafe_allow_html=True)
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")

            if submitted:
                if not email or not password:
                    st.error("Please enter your credentials.")
                else:
                    valid, user_info = authenticate(email, password)
                    if valid:
                        st.session_state.authenticated = True
                        st.session_state.user_email = user_info['email']
                        st.session_state.user_name = user_info['name']
                        st.session_state.user_role = user_info['role']
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")


if not st.session_state.authenticated:
    show_login()
    st.stop()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn3 = torch.nn.BatchNorm1d(hidden)
        self.classifier = torch.nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.bn3(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

HF_REPO = 'Dinal-Jayathilake/graphsentry-artefacts'
THRESHOLD = 0.75
BLOCKSTREAM_API = 'https://blockstream.info/api'


def resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def get_artefact(filename, *local_candidates):
    """Try local paths first, then download from Hugging Face Hub."""
    path = resolve_path(*local_candidates)
    if path:
        return path
    return hf_hub_download(repo_id=HF_REPO, filename=filename)


@st.cache_resource
def load_system():
    # --- Model (always needed) ---
    model_path = get_artefact(
        'model_a.pth',
        'src/models/model_a.pth', 'src/models/final_mvp.pth',
        'models/model_a.pth', 'models/final_mvp.pth',
    )
    if not model_path:
        return None, None, None, None

    # --- Try full-scale pre-computed fingerprints first ---
    try:
        npz_path = get_artefact('fullscale_fingerprints.npz')
        npz = np.load(npz_path)
        fp_array = npz['fingerprints']   # (N, 8)
        risk_scores = npz['risk_scores']  # (N,)
        labels = npz['labels']            # (N,)

        # Infer input dimension from model weights
        state = torch.load(model_path, map_location='cpu')
        in_dim = state['conv1.lin.weight'].shape[1]
        model = GNNClassifier(in_channels=in_dim)
        model.load_state_dict(state)
        model.eval()

        df = pd.DataFrame({
            'id': np.arange(len(risk_scores)),
            'prob_illicit': risk_scores,
            'label': labels.astype(int),
            'nodes': npz['nodes'],
            'edges': npz['edges'],
        })
        df['risk'] = pd.cut(df['prob_illicit'], bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
        df['ground_truth'] = df['label'].map({0: 'Licit', 1: 'Illicit'})

        fingerprints = [fp_array[i] for i in range(len(fp_array))]
        return model, None, df, fingerprints
    except Exception:
        pass

    # --- Fallback: compute from demo_data.pt ---
    data_path = get_artefact('demo_data.pt', 'src/data/demo_data.pt', 'data/demo_data.pt')
    if not data_path:
        return None, None, None, None

    dataset = torch.load(data_path, weights_only=False, map_location='cpu')

    sample_dim = dataset[0].x.size(1)
    if sample_dim == 43:
        for d in dataset:
            deg = degree(d.edge_index[1], d.x.size(0), dtype=torch.float)
            deg = torch.log(deg + 1).view(-1, 1)
            d.x = torch.cat([d.x, deg], dim=1)
        sample_dim = 44

    model = GNNClassifier(in_channels=sample_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    predictions = []
    fingerprints = []
    for i, data in enumerate(dataset):
        with torch.no_grad():
            batch_idx = torch.zeros(data.x.size(0), dtype=torch.long)
            out = model(data.x, data.edge_index, batch_idx)
            prob = F.softmax(out, dim=1)[0, 1].item()

        G = to_networkx(data, to_undirected=True)
        fp = compute_fingerprint(G)
        fingerprints.append(fp)

        predictions.append({
            'id': i, 'prob_illicit': prob, 'label': data.y.item(),
            'nodes': data.num_nodes, 'edges': data.num_edges,
        })

    df = pd.DataFrame(predictions)
    df['risk'] = pd.cut(df['prob_illicit'], bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
    df['ground_truth'] = df['label'].map({0: 'Licit', 1: 'Illicit'})

    return model, dataset, df, fingerprints


# ---------------------------------------------------------------------------
# Structural fingerprinting
# ---------------------------------------------------------------------------

def compute_fingerprint(G):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    if n == 0:
        return np.zeros(8)

    degrees = [d for _, d in G.degree()]
    deg_arr = np.array(degrees, dtype=float)

    return np.array([
        n, e,
        e / max(n, 1),
        deg_arr.mean() if len(deg_arr) > 0 else 0,
        deg_arr.std() if len(deg_arr) > 0 else 0,
        deg_arr.max() if len(deg_arr) > 0 else 0,
        nx.density(G),
        nx.average_clustering(G) if n >= 3 else 0,
    ])


def cosine_similarity(a, b):
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_similar_subgraphs(query_fp, fingerprints, df, top_k=5):
    sims = [(i, cosine_similarity(query_fp, fp)) for i, fp in enumerate(fingerprints)]
    sims.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, sim in sims[:top_k]:
        row = df.iloc[idx]
        results.append({
            'case_id': idx, 'similarity': sim, 'risk_score': row['prob_illicit'],
            'ground_truth': row['ground_truth'], 'nodes': row['nodes'], 'edges': row['edges'],
        })
    return results


# ---------------------------------------------------------------------------
# Blockstream API
# ---------------------------------------------------------------------------

def fetch_address_graph(address, max_txs=25):
    try:
        resp = requests.get(f'{BLOCKSTREAM_API}/address/{address}/txs', timeout=10)
        resp.raise_for_status()
        txs = resp.json()[:max_txs]
    except Exception as e:
        return None, str(e)

    G = nx.DiGraph()
    G.add_node(address, node_type='target')

    for tx in txs:
        for vin in tx.get('vin', []):
            addr = vin.get('prevout', {}).get('scriptpubkey_address')
            if addr:
                G.add_node(addr, node_type='address')
        for vout in tx.get('vout', []):
            addr = vout.get('scriptpubkey_address')
            if addr:
                G.add_node(addr, node_type='address')

        inputs = [v.get('prevout', {}).get('scriptpubkey_address') for v in tx.get('vin', [])]
        outputs = [v.get('scriptpubkey_address') for v in tx.get('vout', [])]
        for src in filter(None, inputs):
            for dst in filter(None, outputs):
                G.add_edge(src, dst)

        time.sleep(0.1)

    return G, None


def fetch_tx_graph(txid):
    try:
        resp = requests.get(f'{BLOCKSTREAM_API}/tx/{txid}', timeout=10)
        resp.raise_for_status()
        tx = resp.json()
    except Exception as e:
        return None, str(e)

    G = nx.DiGraph()
    inputs = [v.get('prevout', {}).get('scriptpubkey_address') for v in tx.get('vin', [])]
    outputs = [v.get('scriptpubkey_address') for v in tx.get('vout', [])]

    for addr in filter(None, inputs):
        G.add_node(addr, node_type='input')
    for addr in filter(None, outputs):
        G.add_node(addr, node_type='output')
    for src in filter(None, inputs):
        for dst in filter(None, outputs):
            G.add_edge(src, dst)

    return G, None


# ---------------------------------------------------------------------------
# Graph visualisation
# ---------------------------------------------------------------------------

def plot_graph_nx(G, target_address=None):
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(height=300, annotations=[dict(text="No data", showarrow=False)])
        return fig

    G_u = G.to_undirected() if G.is_directed() else G
    pos = nx.spring_layout(G_u, seed=42, k=2.0/max(1, G.number_of_nodes()**0.5))

    edge_x, edge_y = [], []
    for u, v in G_u.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    node_x, node_y, node_colors, node_sizes, node_text = [], [], [], [], []
    for n in G_u.nodes():
        node_x.append(pos[n][0]); node_y.append(pos[n][1])
        deg = G_u.degree(n)
        node_sizes.append(10 + 15 * min(deg, 10))
        if n == target_address:
            node_colors.append('#ef4444')
        else:
            node_colors.append('#fafafa' if deg > 2 else '#71717a')
        short = n[:8] + '...' + n[-4:] if isinstance(n, str) and len(n) > 16 else str(n)
        node_text.append(f"{short}<br>Degree: {deg}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                             line=dict(width=0.6, color='#3f3f46'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers',
                             marker=dict(size=node_sizes, color=node_colors,
                                         line=dict(width=0.5, color='#27272a')),
                             text=node_text, hoverinfo='text'))
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      plot_bgcolor='#09090b', paper_bgcolor='#09090b', height=420,
                      font=dict(color='#a1a1aa'))
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def risk_label(score):
    if score >= THRESHOLD:
        return 'High'
    elif score >= 0.3:
        return 'Medium'
    return 'Low'


def render_verdict(score):
    if score >= THRESHOLD:
        st.markdown(f'<div class="verdict-high">HIGH RISK — {score*100:.1f}% estimated illicit probability</div>',
                    unsafe_allow_html=True)
    elif score >= 0.3:
        st.markdown(f'<div class="verdict-medium">MEDIUM RISK — {score*100:.1f}% estimated illicit probability</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-low">LOW RISK — {score*100:.1f}% estimated illicit probability</div>',
                    unsafe_allow_html=True)


def add_to_history(address, risk_score, nodes, edges):
    entry = {
        'address': address,
        'risk_score': risk_score,
        'risk_level': risk_label(risk_score),
        'nodes': nodes,
        'edges': edges,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    st.session_state.investigation_history.insert(0, entry)
    # Keep last 50
    st.session_state.investigation_history = st.session_state.investigation_history[:50]


def generate_report(address, risk_score, matches, props):
    """Generate a CSV-formatted investigation report."""
    lines = [
        f"GraphSentry Investigation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Subject: {address}",
        f"Risk Score: {risk_score*100:.1f}%",
        f"Risk Level: {risk_label(risk_score)}",
        f"",
        f"--- Subgraph Properties ---",
    ]
    for k, v in props.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("--- Matched Known Patterns ---")
    lines.append("Case ID,Similarity,Risk Score,Classification,Nodes,Edges")
    for m in matches:
        lines.append(f"{m['case_id']},{m['similarity']:.4f},{m['risk_score']:.4f},"
                     f"{m['ground_truth']},{m['nodes']},{m['edges']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load system
# ---------------------------------------------------------------------------

model, dataset, df, fingerprints = load_system()

if model is None:
    st.error("System initialisation failed. Unable to load model or reference data.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-logo">GraphSentry</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Dashboard", "Investigate", "Watchlist"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
    st.caption(f"Signed in as **{st.session_state.get('user_email', 'analyst')}**")
    st.caption(f"Investigations: {len(st.session_state.investigation_history)}")
    st.caption(f"Watchlist: {len(st.session_state.watchlist)} addresses")

    st.markdown("")
    if st.button("Sign out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.investigation_history = []
        st.session_state.watchlist = []
        st.rerun()


# ===========================================================================
# Page transition
# ===========================================================================

if page != st.session_state.current_page:
    st.session_state.current_page = page
    with st.container():
        st.markdown(
            '<div style="display:flex;justify-content:center;align-items:center;'
            'height:50vh;"><div style="color:#52525b;font-size:0.85rem;">Loading...</div></div>',
            unsafe_allow_html=True,
        )
    time.sleep(0.3)
    st.rerun()


# ===========================================================================
# PAGE: DASHBOARD
# ===========================================================================

if page == "Dashboard":
    st.markdown('<div class="page-header">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Overview of investigation activity and watchlist status.</div>',
                unsafe_allow_html=True)

    history = st.session_state.investigation_history
    watchlist = st.session_state.watchlist

    # --- Metrics row ---
    col1, col2, col3, col4 = st.columns(4)

    n_inv = len(history)
    n_inv_high = sum(1 for h in history if h['risk_level'] == 'High')
    n_inv_med = sum(1 for h in history if h['risk_level'] == 'Medium')
    n_inv_low = sum(1 for h in history if h['risk_level'] == 'Low')

    col1.metric("Total Investigations", f"{n_inv}")
    col2.metric("High Risk Flagged", f"{n_inv_high}")
    col3.metric("Watchlist", f"{len(watchlist)}")
    col4.metric("Active Alerts", f"{sum(1 for w in watchlist if w['risk_level'] == 'High')}")

    st.markdown("")

    col_chart, col_history = st.columns([1, 1])

    # --- Risk distribution of investigations ---
    with col_chart:
        st.markdown('<div class="section-header">Investigation Risk Breakdown</div>',
                    unsafe_allow_html=True)

        if not history:
            st.markdown(
                '<div class="empty-state">'
                ''
                '<div class="empty-state-title">No data yet</div>'
                '<div class="empty-state-desc">Risk distribution will appear as you investigate addresses.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            fig_dist = go.Figure(data=[go.Pie(
                labels=['Low', 'Medium', 'High'],
                values=[n_inv_low, n_inv_med, n_inv_high],
                hole=0.55,
                marker=dict(colors=['#22c55e', '#eab308', '#ef4444'],
                            line=dict(color='#09090b', width=2)),
                textinfo='label+value',
                textfont=dict(size=13, color='#fafafa'),
                hoverinfo='label+percent',
            )])
            fig_dist.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20), height=300,
                plot_bgcolor='#09090b', paper_bgcolor='#09090b',
                font=dict(family='Inter', color='#a1a1aa'),
                annotations=[dict(text=f'{n_inv}', x=0.5, y=0.5,
                                  font=dict(size=28, color='#fafafa', family='Inter'),
                                  showarrow=False)],
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # --- Recent investigations ---
    with col_history:
        st.markdown('<div class="section-header">Recent Investigations</div>', unsafe_allow_html=True)

        if not history:
            st.markdown(
                '<div class="empty-state">'
                ''
                '<div class="empty-state-title">No investigations yet</div>'
                '<div class="empty-state-desc">Navigate to Investigate to analyse a Bitcoin address.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for entry in history[:8]:
                addr = entry['address']
                short = addr[:12] + '...' + addr[-6:] if len(addr) > 20 else addr
                level = entry['risk_level']
                color = '#ef4444' if level == 'High' else '#eab308' if level == 'Medium' else '#22c55e'
                st.markdown(
                    f'<div class="history-row">'
                    f'<div><div class="history-addr">{short}</div>'
                    f'<div class="history-meta">{entry["timestamp"]} · {entry["nodes"]} addresses</div></div>'
                    f'<div style="color:{color};font-weight:600;font-size:0.85rem;">{level}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ===========================================================================
# PAGE: INVESTIGATE
# ===========================================================================

elif page == "Investigate":
    st.markdown('<div class="page-header">Investigate Address</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Analyse a Bitcoin address or transaction by building its '
                'transaction graph from live blockchain data and matching it against known patterns.</div>',
                unsafe_allow_html=True)

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        query = st.text_input(
            "Bitcoin address or transaction ID",
            placeholder="Enter a Bitcoin address (e.g. bc1q..., 3ADPk...) or a 64-character transaction ID",
            label_visibility="collapsed",
        )
    with col_btn:
        go_btn = st.button("Analyse", type="primary", use_container_width=True)

    if go_btn and query.strip():
        query = query.strip()
        is_txid = len(query) == 64 and all(c in '0123456789abcdef' for c in query.lower())

        with st.spinner("Retrieving transaction data from blockchain..."):
            if is_txid:
                G, err = fetch_tx_graph(query)
            else:
                G, err = fetch_address_graph(query)

        if err:
            st.error(f"Unable to retrieve data: {err}")
        elif G is None or G.number_of_nodes() == 0:
            st.warning("No transaction data found for this input.")
        else:
            G_undirected = G.to_undirected()
            query_fp = compute_fingerprint(G_undirected)
            matches = find_similar_subgraphs(query_fp, fingerprints, df, top_k=5)

            total_weight = sum(m['similarity'] for m in matches)
            estimated_risk = (sum(m['risk_score'] * m['similarity'] for m in matches) / total_weight
                              if total_weight > 0 else 0.5)

            # Record in history
            add_to_history(query, estimated_risk, G.number_of_nodes(), G.number_of_edges())

            # Verdict
            render_verdict(estimated_risk)

            props = {
                'Addresses': str(G.number_of_nodes()),
                'Connections': str(G.number_of_edges()),
                'Network density': f"{nx.density(G_undirected):.4f}",
                'Avg connections': f"{np.mean([d for _, d in G_undirected.degree()]):.2f}",
                'Max connections': str(max((d for _, d in G_undirected.degree()), default=0)),
                'Clustering': f"{nx.average_clustering(G_undirected):.4f}" if G_undirected.number_of_nodes() >= 3 else "N/A",
                'Sub-networks': str(nx.number_connected_components(G_undirected)),
            }

            col_graph, col_details = st.columns([3, 2])

            with col_graph:
                st.markdown('<div class="section-header">Transaction Graph</div>',
                            unsafe_allow_html=True)
                target = query if not is_txid else None
                fig = plot_graph_nx(G, target_address=target)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Highlighted node = queried address. Size = connection count. "
                           f"{G.number_of_nodes()} addresses, {G.number_of_edges()} connections.")

                col_act1, col_act2 = st.columns(2)
                with col_act1:
                    report = generate_report(query, estimated_risk, matches, props)
                    st.download_button(
                        "Export Report",
                        data=report,
                        file_name=f"graphsentry_report_{query[:16]}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with col_act2:
                    already_watched = any(w['address'] == query for w in st.session_state.watchlist)
                    if already_watched:
                        st.button("On Watchlist", disabled=True, use_container_width=True)
                    else:
                        if st.button("Add to Watchlist", use_container_width=True):
                            st.session_state.watchlist.append({
                                'address': query,
                                'risk_score': estimated_risk,
                                'risk_level': risk_label(estimated_risk),
                                'added': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                'nodes': G.number_of_nodes(),
                            })
                            st.toast("Address added to watchlist")
                            st.rerun()

            with col_details:
                st.markdown('<div class="section-header">Network Properties</div>',
                            unsafe_allow_html=True)
                for label, value in props.items():
                    st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span>'
                                f'<span class="stat-value">{value}</span></div>', unsafe_allow_html=True)

                st.markdown("")
                st.markdown('<div class="section-header">Matched Known Patterns</div>',
                            unsafe_allow_html=True)

                for m in matches:
                    gt_color = '#ef4444' if m['ground_truth'] == 'Illicit' else '#22c55e'
                    st.markdown(f"""<div class="match-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-weight:600;color:#fafafa;">Case #{m['case_id']}</span>
                            <span style="font-size:0.75rem;color:#52525b;">{m['similarity']*100:.0f}% match</span>
                        </div>
                        <div style="font-size:0.8rem;color:#71717a;margin-top:0.25rem;">
                            Risk: {m['risk_score']*100:.1f}% · <span style="color:{gt_color};">{m['ground_truth']}</span> · {m['nodes']} addresses, {m['edges']} connections
                        </div>
                    </div>""", unsafe_allow_html=True)


    elif go_btn:
        st.warning("Please enter a Bitcoin address or transaction ID.")

    elif not query:
        st.markdown(
            '<div class="empty-state">'
            ''
            '<div class="empty-state-title">Enter an address to begin</div>'
            '<div class="empty-state-desc">Paste a Bitcoin address or transaction ID above to build and '
            'analyse its transaction network against known illicit patterns.</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# PAGE: WATCHLIST
# ===========================================================================

elif page == "Watchlist":
    st.markdown('<div class="page-header">Watchlist</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Flagged addresses for ongoing monitoring and review.</div>',
                unsafe_allow_html=True)

    if not st.session_state.watchlist:
        st.markdown(
            '<div class="empty-state">'
            ''
            '<div class="empty-state-title">Watchlist is empty</div>'
            '<div class="empty-state-desc">Investigate an address and click "Add to Watchlist" to '
            'flag it for monitoring.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        wl = st.session_state.watchlist

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        n_w_high = sum(1 for w in wl if w['risk_level'] == 'High')
        n_w_med = sum(1 for w in wl if w['risk_level'] == 'Medium')
        n_w_low = sum(1 for w in wl if w['risk_level'] == 'Low')
        col1.metric("Watched Addresses", len(wl))
        col2.metric("High Risk", n_w_high)
        col3.metric("Medium / Low", f"{n_w_med} / {n_w_low}")

        st.markdown("")

        # Watchlist table
        to_remove = None
        for i, w in enumerate(wl):
            addr = w['address']
            short = addr[:16] + '...' + addr[-6:] if len(addr) > 24 else addr
            level = w['risk_level']
            color = '#ef4444' if level == 'High' else '#eab308' if level == 'Medium' else '#22c55e'

            col_addr, col_risk, col_date, col_action = st.columns([4, 1, 1.5, 1])
            with col_addr:
                st.markdown(f'<div style="font-family:monospace;font-size:0.85rem;color:#fafafa;'
                            f'padding-top:0.5rem;">{short}</div>', unsafe_allow_html=True)
            with col_risk:
                st.markdown(f'<div style="color:{color};font-weight:600;font-size:0.85rem;'
                            f'padding-top:0.5rem;">{level}</div>', unsafe_allow_html=True)
            with col_date:
                st.markdown(f'<div style="color:#71717a;font-size:0.8rem;'
                            f'padding-top:0.5rem;">{w["added"]}</div>', unsafe_allow_html=True)
            with col_action:
                if st.button("Remove", key=f"rm_watch_{i}", use_container_width=True):
                    to_remove = i

            st.markdown('<hr style="border:none;border-top:1px solid #27272a;margin:0.25rem 0;">',
                        unsafe_allow_html=True)

        if to_remove is not None:
            st.session_state.watchlist.pop(to_remove)
            st.rerun()

        # Export watchlist
        st.markdown("")
        wl_data = pd.DataFrame(st.session_state.watchlist)
        csv = wl_data.to_csv(index=False)
        st.download_button(
            "Export Watchlist",
            data=csv,
            file_name=f"graphsentry_watchlist_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

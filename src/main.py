import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx, degree
import networkx as nx
import matplotlib.pyplot as plt
import os


class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn3 = torch.nn.BatchNorm1d(hidden)
        self.classifier = torch.nn.Linear(hidden, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.bn3(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)


def add_degree_feature(data):
    deg = degree(data.edge_index[1], data.x.size(0), dtype=torch.float)
    deg = torch.log(deg + 1).view(-1, 1)
    data.x = torch.cat([data.x, deg], dim=1)
    return data


@st.cache_resource
def load_system():
    data_path = 'data/demo_data.pt'
    model_path = 'models/final_mvp.pth'

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        return None, None

    dataset = torch.load(data_path, weights_only=False)
    input_dim = dataset[0].x.size(1)

    model = GNNClassifier(in_channels=input_dim, hidden=128)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, dataset

# Page config
st.set_page_config(page_title="GraphSentry", layout="wide")
st.title("GraphSentry")
st.markdown("Anti-Money Laundering Detection System")

model, dataset = load_system()

if model is None or dataset is None:
    st.error("Missing data or model files. Check paths.")
    st.stop()

# Sidebar
st.sidebar.header("Case Selection")
case_labels = [f"Case #{i} ({'Illicit' if d.y.item() == 1 else 'Licit'})" for i, d in enumerate(dataset)]
selected = st.sidebar.selectbox("Select a transaction graph", case_labels)
case_idx = case_labels.index(selected)

data = dataset[case_idx].clone()

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Analysis")
    
    with torch.no_grad():
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        out = model(data.x, data.edge_index, data.batch)
        probs = F.softmax(out, dim=1)
        prob_illicit = probs[0][1].item()
    
    # Threshold from notebook evaluation
    threshold = 0.4
    
    st.metric("Illicit Probability", f"{prob_illicit * 100:.2f}%")
    
    if prob_illicit >= threshold:
        st.error("VERDICT: HIGH RISK")
    else:
        st.success("VERDICT: LOW RISK")

    st.markdown("---")
    st.write(f"Nodes: {data.num_nodes}")
    st.write(f"Edges: {data.num_edges}")
    st.write(f"Ground Truth: {'Illicit' if data.y.item() == 1 else 'Licit'}")

with col2:
    st.subheader("Transaction Graph")
    
    G = to_networkx(data, to_undirected=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pos = nx.spring_layout(G, seed=42)
    node_degrees = [val for (_, val) in G.degree()]
    
    nx.draw(
        G, pos, ax=ax,
        node_size=150,
        node_color=node_degrees,
        cmap=plt.cm.plasma,
        edge_color="gray",
        alpha=0.9,
        with_labels=False
    )
    
    st.pyplot(fig)
    st.caption("Node brightness indicates connection count (potential hubs)")
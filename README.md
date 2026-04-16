# GraphSentry: AML Forensic Inspector

GraphSentry is a Graph Neural Network (GNN) framework designed to detect money laundering patterns on the Bitcoin blockchain. It classifies entire transaction subgraphs as "Licit" or "Illicit" using structural pattern recognition, and provides an interactive forensic analyst dashboard for investigating live Bitcoin addresses.

**Live Demo:** [graphsentry-vdwke5uv9ah3sbklr4b4u9.streamlit.app](https://graphsentry-vdwke5uv9ah3sbklr4b4u9.streamlit.app/)

## Features

- **Graph-Based Detection**: Analyses Bitcoin transaction networks as graphs using a 2-layer GCN with residual connections and global max pooling
- **Live Address Investigation**: Fetches real transaction data from the Blockstream API and scores addresses against 121K+ known patterns
- **Structural Fingerprinting**: Computes 8-dimensional graph fingerprints and finds similar subgraphs via cosine similarity
- **Interactive Dashboard**: Streamlit-based dark-themed UI with investigation history, risk breakdowns, and a watchlist
- **Persistent Sessions**: Stay signed in across page reloads
- **Export**: Download investigation reports and watchlists as CSV

## Quick Start Guide

This project uses [uv](https://github.com/astral-sh/uv) for ultra-fast dependency management. You do not need to manually install Python or manage virtual environments — uv handles everything.

### Step 1: Install uv

If you don't have uv installed, open your terminal and run one of the following:

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> **Note**: Close and reopen your terminal after installing to ensure the command is recognized.

### Step 2: Navigate to the Project

Open your terminal and navigate into the project folder:

```bash
cd path/to/GraphSentry
```

### Step 3: Run the Application

Run this single command to automatically download Python, install all required libraries (PyTorch, PyTorch Geometric, Streamlit), and launch the app:

```bash
uv run streamlit run src/app.py
```

> **Note**: The first run may take a minute to download dependencies. Model weights and reference data are automatically fetched from Hugging Face on first launch.

## Project Structure

```
GraphSentry/
├── README.md                          # This guide
├── pyproject.toml                     # Dependency configuration
├── uv.lock                            # Locked versions for reproducibility
├── .gitignore                         # Git ignore rules
├── .python-version                    # Python version specification
├── .streamlit/
│   └── config.toml                    # Streamlit theme and server settings
└── src/
    ├── app.py                         # Main application — forensic analyst dashboard
    ├── main.py                        # Initial MVP prototype (not actively used)
    └── notebooks/
        └── GraphSentry_Complete.ipynb # Full experiment pipeline (data, training, ablations, evaluation)
```

> `src/main.py` was the initial MVP used during early development. It loads 50 pre-packaged test cases with a simple selectbox UI. The full application is `src/app.py`, which adds authentication, live blockchain lookups, structural fingerprinting against 121K+ reference subgraphs, and the complete forensic dashboard.
# GraphSentry: AML Forensic Inspector

GraphSentry is a Graph Neural Network (GNN) framework designed to detect money laundering patterns on the Bitcoin blockchain. This prototype demonstrates the system's ability to classify entire transaction subgraphs as "Licit" or "Illicit" using structural pattern recognition.

## Features

- **Graph-Based Detection**: Analyzes Bitcoin transaction networks as graphs
- **Deep Learning**: 3-layer GCN with batch normalization for robust pattern recognition
- **Interactive Dashboard**: Streamlit-based visualization of transaction graphs and risk assessments
- **Real-Time Analysis**: Instant classification of 50 pre-loaded test cases

## Quick Start Guide

This project uses [uv](https://github.com/astral-sh/uv) for ultra-fast dependency management. You do not need to manually install Python or manage virtual environments—uv handles everything.

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
uv run streamlit run src/main.py
```

> **Note**: The first run may take a minute to download dependencies.

## Project Structure

```
GraphSentry/
├── README.md                          # This guide
├── pyproject.toml                     # Dependency configuration
├── uv.lock                            # Locked versions for reproducibility
├── .gitignore                         # Git ignore rules
├── .python-version                    # Python version specification
└── src/
    ├── main.py                        # Streamlit dashboard application
    ├── data/
    │   └── demo_data.pt               # 50 pre-packaged test cases
    ├── models/
    │   └── final_mvp.pth              # Trained GNN model weights
    └── notebooks/
        └── 01_preprocessing.ipynb     # Data preprocessing notebook
```
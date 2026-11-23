# SmartEdge AI

**Think Smart. Compute Local. Build Responsible AI.**

SmartEdge AI is a web platform for responsible, resource-efficient AI. Instead of sending every task to a large cloud model, it asks where and how a task should run.

## Problem
Cloud-based AI can waste computing power, expose private data, depend on a network connection, use oversized models for simple tasks, treat groups unfairly and hide how decisions are made.

## Features
| Module | What it does |
|---|---|
| Edge AI Engine | Runs predictions locally with scikit-learn and times each request |
| Resource Monitor | Tracks real CPU, RAM and app memory with psutil |
| Privacy Guardian | Detects and masks emails, phones, Aadhaar, PAN and card numbers |
| Fairness Checker | Compares accuracy and approval rates across groups (generated data or your own CSV) |
| Smart Optimizer | Benchmarks small, medium and large models and recommends the smallest accurate one |
| Analytics | Logs each session and exports a CSV report |

## Quick start
```bash
git clone https://github.com/YOUR-USERNAME/smartedge-ai.git
cd smartedge-ai
pip install -r requirements.txt
streamlit run app.py
```
Requires Python 3.9 or newer.

The standalone web UI prototype is in `prototype/index.html`. Open it in Chrome or Edge.

## Measured vs simulated
| Item | Status |
|---|---|
| CPU %, RAM %, app memory | Measured |
| Inference time and CPU time per request | Measured |
| Model size and accuracy | Measured, on generated data |
| Fairness metrics | Measured, on generated or uploaded data |
| Training data | Simulated (template-generated reviews) |
| Cloud latency | Simulated (set in the UI) |
| Energy use | Not measured, not claimed |

## Project structure
```
smartedge-ai/
├── app.py            Streamlit backend
├── requirements.txt  Python dependencies
├── prototype/
│   └── index.html    Standalone web UI prototype
├── .gitignore
└── README.md
```

## Tech stack
Python, Streamlit, scikit-learn, pandas, NumPy, psutil.

## Limitations and next steps
- Replace `make_dataset()` in `app.py` with a real labelled dataset.
- Replace the simulated cloud latency with a timed call to a real API.
- Privacy Guardian is rule-based and can miss names and free-text details.
- Add energy measurement with hardware-level tools before making any energy claims.

## Author
Anbuchelvan V, Tech Takers - COMPUTE 50 .

#!/usr/bin/env bash
# Launch the XGBoost + TreeSHAP GUI
cd "$(dirname "$0")"
exec .venv/bin/streamlit run app.py --server.maxUploadSize 1024 "$@"

#!/bin/bash
pip install -r co-po-burt/requirements.txt
streamlit run co-po-burt/app.py --server.port=$PORT --server.address=0.0.0.0








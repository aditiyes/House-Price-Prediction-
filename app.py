import os
import sys

# Entry point wrapper for Streamlit
frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "app.py")
with open(frontend_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(code)

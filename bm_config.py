import os, random, sqlite3
import numpy as np

# Cloud-safe project root: works locally, on GitHub/Streamlit, and on Windows/Linux.
BASE = os.path.dirname(os.path.abspath(__file__))
P = {k: os.path.join(BASE, k) for k in ["data", "models", "db", "skin_data", "outputs", "app"]}
DB_PATH = os.path.join(P["db"], "beautymind.db")
SEED = 42

def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)

def get_conn():
    # check_same_thread=False is useful for Streamlit's rerun/thread model.
    return sqlite3.connect(DB_PATH, check_same_thread=False)

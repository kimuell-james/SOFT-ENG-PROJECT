import pandas as pd

def load_data(path='data/student_data.csv'):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"🚫 File not found at path: {path}")
    except Exception as e:
        raise RuntimeError(f"⚠️ Failed to load data: {e}")

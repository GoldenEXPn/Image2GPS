# preprocess.py
import os
import numpy as np
import pandas as pd

# Backend CSV contract columns (Project A)
_PATH_COLS = ["image_path", "filepath", "image", "path", "file_name"]
_LAT_COLS = ["Latitude", "latitude", "lat"]
_LON_COLS = ["Longitude", "longitude", "lon"]


def _find_col(df_cols, candidates):
    cols = list(df_cols)
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    # fallback: case-insensitive
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _prepare_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    path_col = _find_col(df.columns, _PATH_COLS)
    lat_col = _find_col(df.columns, _LAT_COLS)
    lon_col = _find_col(df.columns, _LON_COLS)

    if path_col is None or lat_col is None or lon_col is None:
        raise KeyError(
            f"CSV missing required columns. Found: {list(df.columns)}. "
            f"Need path in {_PATH_COLS}, lat in {_LAT_COLS}, lon in {_LON_COLS}."
        )

    base_dir = os.path.dirname(os.path.abspath(csv_path))

    paths = []
    for p in df[path_col].astype(str).tolist():
        p = p.strip()
        if not os.path.isabs(p):
            p = os.path.join(base_dir, p)
        paths.append(p)

    y = df[[lat_col, lon_col]].astype(np.float32).values  # raw degrees
    return paths, y


def _parse_hf_spec(spec: str):
    """
    Supported:
      hf://dataset_name
      hf://dataset_name:split
    """
    s = spec.strip()
    if s.startswith("hf://"):
        s = s[len("hf://") :]
    elif s.startswith("hf:"):
        s = s[len("hf:") :]

    split = "train"
    if ":" in s:
        name, split = s.split(":", 1)
        name, split = name.strip(), split.strip()
    else:
        name = s.strip()

    if not name:
        raise ValueError(f"Bad HuggingFace spec: {spec}")
    return name, split


def _prepare_from_hf(spec: str):
    # Lazy import: backend env may not include `datasets` :contentReference[oaicite:6]{index=6}
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HF loading requested but `datasets` is not installed. "
            "Install locally via `pip install datasets`. "
            "For submission, pass a real CSV path."
        ) from e

    ds_name, split = _parse_hf_spec(spec)
    ds = load_dataset(ds_name, split=split)

    # Your dataset schema is: image, latitude, longitude :contentReference[oaicite:7]{index=7}
    if not all(k in ds.column_names for k in ["image", "latitude", "longitude"]):
        raise KeyError(f"Unexpected columns: {ds.column_names}. Expected image/latitude/longitude.")

    X = ds["image"]  # typically PIL Images via HF datasets Image feature
    y = np.stack(
        [
            np.asarray(ds["latitude"], dtype=np.float32),
            np.asarray(ds["longitude"], dtype=np.float32),
        ],
        axis=1,
    )
    return X, y


def prepare_data(csv_path: str):
    """
    Submission contract: csv_path is a CSV on disk :contentReference[oaicite:8]{index=8}
    Local convenience: supports HF spec hf://...:split
    """
    if os.path.isfile(csv_path) and csv_path.lower().endswith(".csv"):
        return _prepare_from_csv(csv_path)

    if csv_path.startswith("hf://") or csv_path.startswith("hf:"):
        return _prepare_from_hf(csv_path)

    # default: treat as CSV path
    return _prepare_from_csv(csv_path)

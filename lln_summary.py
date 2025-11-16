from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option('display.max_colwidth', 160)
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 10)

BASE_DIR = Path("./results").resolve()
DIRS = ["poisson", "heat"]

print(f"Reading results from: {BASE_DIR}")

files_map = {
    "baseline": "Baseline",
    "lln1": "LLN(1)",
    "lln1e-16": "LLN(1e-16)",
    "multiadam_baseline": "MultiAdam Baseline",
    "multiadam_lln1": "MultiAdam LLN(1)",
    "multiadam_lln1e-16": "MultiAdam LLN(1e-16)",
}

metrics = {
    "mse": "MSE",
    "mxe": "Max error",
}


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df


def _make_metric_tables(files_map: dict[str, str], metric_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_pdes: set[str] = set()
    mean_rows: dict[str, pd.Series] = {}
    std_rows: dict[str, pd.Series] = {}

    for method_key, label in files_map.items():
        mean_parts = []
        std_parts = []

        for subdir in DIRS:
            path = BASE_DIR / subdir / f"{subdir}-{method_key}.csv"
            df = _read_csv_safe(path)
            if df is None:
                continue
            if "pde" not in df.columns or metric_col not in df.columns:
                continue

            df = df.set_index("pde")
            mean_s = df[metric_col]
            mean_parts.append(mean_s)
            all_pdes.update(mean_s.index.tolist())

            std_col = f"{metric_col}_std"
            if std_col in df.columns:
                std_s = df[std_col]
                std_parts.append(std_s)

        if mean_parts:
            mean_concat = pd.concat(mean_parts, axis=0)
            mean_rows[label] = mean_concat
        else:
            mean_rows[label] = pd.Series(dtype=float)

        if std_parts:
            std_concat = pd.concat(std_parts, axis=0)
            std_rows[label] = std_concat
        else:
            std_rows[label] = pd.Series(dtype=float)

    # pdes_sorted = sorted(all_pdes)
    pdes_sorted = list(all_pdes)
    methods_order = [files_map[k] for k in files_map]

    mean_table = pd.DataFrame(index=methods_order, columns=pdes_sorted, dtype=float)
    std_table = pd.DataFrame(index=methods_order, columns=pdes_sorted, dtype=float)

    for label, s in mean_rows.items():
        mean_table.loc[label, s.index] = s.values

    for label, s in std_rows.items():
        std_table.loc[label, s.index] = s.values

    return mean_table, std_table


def to_pretty(mean_df: pd.DataFrame, std_df: pd.DataFrame | None = None) -> pd.DataFrame:
    out = mean_df.copy().astype(object)

    if std_df is None:
        for r in mean_df.index:
            for c in mean_df.columns:
                v = mean_df.loc[r, c]
                if pd.isna(v):
                    out.loc[r, c] = np.nan
                else:
                    out.loc[r, c] = float(f"{v:.3g}")
        return out

    for r in mean_df.index:
        for c in mean_df.columns:
            m = mean_df.loc[r, c]
            if pd.isna(m):
                out.loc[r, c] = np.nan
                continue

            s = std_df.loc[r, c] if c in std_df.columns else np.nan
            if pd.isna(s) or s == 0:
                out.loc[r, c] = float(f"{m:.3g}")
            else:
                out.loc[r, c] = f"{m:.3g} ± {s:.2g}"

    return out


if __name__ == "__main__":
    mse_mean, mse_std = _make_metric_tables(files_map, "mse")
    mxe_mean, mxe_std = _make_metric_tables(files_map, "mxe")

    print("=== MSE ===")
    print(to_pretty(mse_mean, mse_std))

    print("\n=== Max error (mxe) ===")
    print(to_pretty(mxe_mean, mxe_std))

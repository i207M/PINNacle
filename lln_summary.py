import pandas as pd
from pathlib import Path
import numpy as np


pd.set_option('display.max_colwidth', 160)
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 10)



RESULTS_PATH = Path("./results").resolve()
print(f'Reading results from: {RESULTS_PATH}')

files_map = {
    "baseline.csv":              "Baseline",
    "lln1.csv":                  "LLN(1)",
    "lln1e-16.csv":              "LLN(1e-16)",
    "multiadam_baseline.csv":    "MultiAdam Baseline",
    "multiadam_lln1.csv":        "MultiAdam LLN(1)",
    "multiadam_lln1e-16.csv":    "MultiAdam LLN(1e-16)",
}


metrics = {
    "mse": "MSE",
    "mxe": "Max error",
}

def _fmt(x):
    if pd.isna(x):
        return np.nan
    return float(f"{x:.3g}")

def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # usuń ewentualne "Unnamed: 0" itp.
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df

def _make_metric_table(files_map, metric_col: str) -> pd.DataFrame:
    # union of all pdes
    all_pdes = set()
    rows = {}

    for fname, label in files_map.items():
        df = _read_csv_safe(RESULTS_PATH / fname)
        if df is not None:
            # sanity: columns must contain 'pde' and the given metric
            if "pde" not in df.columns or metric_col not in df.columns:
                # if there is no such column, treat as no data
                rows[label] = pd.Series(dtype=float)
                continue

            # take only pde and metric, reduce to series pde->metric
            s = df.set_index("pde")[metric_col]
            s = s.map(_fmt)
            rows[label] = s
            all_pdes.update(s.index.tolist())
        else:
            # no file
            rows[label] = pd.Series(dtype=float)

    # build a table: rows = methods, columns = pdes
    pdes_sorted = sorted(all_pdes)
    methods_order = [files_map[f] for f in files_map]

    table = pd.DataFrame(index=methods_order, columns=pdes_sorted, dtype=float)

    for label, s in rows.items():
        table.loc[label, s.index] = s.values

    return table




if __name__ == '__main__':

    # === budowanie tabel ===
    mse_table = _make_metric_table(files_map, "mse")
    mxe_table = _make_metric_table(files_map, "mxe")

    # na podgląd (opcjonalnie): 3 cyfry zn., NaN jako '—'
    def to_pretty(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        return out

    print("=== MSE ===")
    print(to_pretty(mse_table))
    print("\n=== Max error (mxe) ===")
    print(to_pretty(mxe_table))

    # === eksport do LaTeX ===
    # ustaw ładny zapis NaN jako '—'
    latex_kwargs = dict(na_rep="—", escape=True)

    # jeżeli chcesz mieć wymuszone formatowanie liczb w LaTeX (np. 3 cyfry zn.),
    # możesz użyć float_format, ale to dotyczy całej tabeli:
    # float_format=lambda x: f"{x:.3g}"

    with open("mse_table.tex", "w", encoding="utf-8") as f:
        f.write(mse_table.to_latex(**latex_kwargs, float_format=lambda x: f"{x:.3g}"))

    with open("max_error_table.tex", "w", encoding="utf-8") as f:
        f.write(mxe_table.to_latex(**latex_kwargs, float_format=lambda x: f"{x:.3g}"))

    print("\nZapisano: mse_table.tex, max_error_table.tex")

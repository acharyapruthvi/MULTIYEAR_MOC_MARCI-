import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import glob
import re
import os

# ==========================Functions==========================
def obs_prefix(path):
    """'B11_Ls_3100-3119_rgb_cluster_coords.csv' -> 'B11'."""
    m = _PREFIX_RE.match(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path).split("_")[0]

def load_conversion(path):
    """
    Build PREFIX -> {'my': int|None} from MARCI_DATE_CONVERSION.csv, whose
    columns are: Earth Month, Date, MY, Ls

        Earth Month,Date,MY,Ls
        P01,Nov-06,28,Ls 129.0

    Keys are upper-cased and stripped so 'p01' in a filename still matches
    'P01' in the table. CONV_PREFIX_COL / CONV_MY_COL override the lookup;
    otherwise the columns are matched by name, case- and space-insensitively.
    """
    if not path or not os.path.exists(path):
        if path:
            print(f"  [note] {path} not found; MY will be blank")
        return {}

    df = pd.read_csv(path)
    # normalise "Earth Month" / "earth_month" / " MY " to a common key
    cols = {re.sub(r"[\s_]+", " ", str(c)).strip().lower(): c
            for c in df.columns}

    pcol = CONV_PREFIX_COL
    if pcol is None:
        for key in ("earth month", "month", "prefix", "observation", "obs",
                    "code", "id", "file"):
            pcol = next((cols[k] for k in cols if key in k), None)
            if pcol is not None:
                break
    mcol = CONV_MY_COL
    if mcol is None:
        mcol = next((cols[k] for k in cols
                     if k in ("my", "mars year", "marsyear", "year")), None)

    if pcol is None or mcol is None:
        print(f"  [note] could not identify the Earth-month / MY columns in "
              f"{path} (columns: {list(df.columns)}); MY will be blank. "
              f"Set CONV_PREFIX_COL and CONV_MY_COL explicitly.")
        return {}

    out = {}
    for _, r in df.iterrows():
        key = str(r[pcol]).strip().upper()
        if not key or key == "NAN":
            continue
        try:
            my = int(float(r[mcol]))
        except (TypeError, ValueError):
            my = None
        out[key] = {"my": my}

    n_my = sum(1 for v in out.values() if v["my"] is not None)
    print(f"  Loaded {len(out)} Earth months from {path} "
          f"(columns '{pcol}' -> '{mcol}'), {n_my} with an MY")
    return out

def lookup_my(conv, prefix):
    return conv.get(str(prefix).strip().upper(), {}).get("my")

def ls_mid_from_name(name):
    m = _LS_RE.search(os.path.basename(str(name)))
    if not m:
        return np.nan
    return ((int(m.group(1)) + int(m.group(2)) + 1) / 20) % 360

def wrap_ls(s, start=None):
    """
    Put Ls on a continuous axis that begins at `start` instead of at 0.
    Returns values in [start, start + 360).  With start=340:
        Ls 340 -> 340,  Ls 359 -> 359,  Ls 0 -> 360,  Ls 70 -> 430.
    Plot against these and relabel the ticks with ls_tick_label().
    Works on a Series or a scalar.
    """
    if start is None:
        start = LS_WINDOW_START
    s = pd.to_numeric(s, errors="coerce")
    return (s - start) % 360 + start

def ls_tick_label(x, _pos=None):
    """Turn a wrapped axis coordinate back into a true Ls label."""
    return f"{x % 360:g}"

def report_outside_window(df, label):
    """Warn about rows that wrap outside the plotted Ls window."""
    lo = LS_WINDOW_START
    hi = LS_WINDOW_START + LS_WINDOW_SPAN
    out = df[(df["Wrapped_Ls"] < lo) | (df["Wrapped_Ls"] > hi)]
    if len(out):
        bins = sorted(out["Ls"].dropna().unique())
        print(f"  [note] {label}: {len(out)} row(s) fall outside the "
              f"Ls {lo:g}-{hi % 360:g} window and will not be drawn "
              f"(true Ls: {bins})")

# ── Climatological average (binned mean across all MYs) ───────────────────────
def climatological_average(data, bin_width=5.0,
                            lo=None, hi=None):
    """
    Bin Wrapped_Ls into fixed bin_width-degree bins running from lo to hi
    and compute the mean avg_major_latitude per bin.
    """
    data = data.copy()
    n_bins = int(round((hi - lo) / bin_width))
    bins = np.linspace(lo, hi, n_bins + 1)
    data["Ls_bin"] = pd.cut(data["Wrapped_Ls"], bins=bins)
    clim_lat = data.groupby("Ls_bin", observed=True)["avg_major_latitude"].mean()
    bin_centers = np.array([iv.mid for iv in clim_lat.index])
    return bin_centers, clim_lat.values

# ==========================Variables==========================
# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------
_LS_RE = re.compile(r"_Ls_(\d+)[-_](\d+)", re.IGNORECASE)
_PREFIX_RE = re.compile(r"^([A-Za-z0-9]+)_Ls_", re.IGNORECASE)
CONV_PREFIX_COL = None  # set to exact column name to override auto-detection
CONV_MY_COL     = None  # set to exact column name to override auto-detection

# --------------------------------------------------------------------------
# Ls axis window: the plot runs from LS_WINDOW_START rightwards for
# LS_WINDOW_SPAN degrees.  340 + 90 -> the axis reads 340 ... 360/0 ... 70.
# Set LS_WINDOW_START = 330.0 / LS_WINDOW_SPAN = 100.0 to also include the
# Ls 331-339 MARCI bins.
# --------------------------------------------------------------------------
LS_WINDOW_START = 340.0
LS_WINDOW_SPAN  = 90.0
LS_TICK_STEP    = 10.0
MY_BIN_WIDTH = 5.0  # degrees per bin for the per-MY Ls average

polar_north_base_dir = "polar_north"
polar_south_base_dir = "polar_south"

moc_north_dir = f"{polar_north_base_dir}/MOC/*.csv"
marci_north_dir = f"{polar_north_base_dir}/MARCI/*.csv"

moc_south_dir = f"{polar_south_base_dir}/MOC/*.csv"
marci_south_dir = f"{polar_south_base_dir}/MARCI/*.csv"

marci_date_conversion_csv = load_conversion("MARCI_DATE_CONVERSION.csv")

# ==========================Import Data==========================
moc_north_files = glob.glob(moc_north_dir)
marci_north_files = glob.glob(marci_north_dir)

moc_north_dataframe = pd.concat([pd.read_csv(f) for f in moc_north_files], ignore_index=True)
marci_north_dataframe = pd.concat([pd.read_csv(f) for f in marci_north_files], ignore_index=True)

marci_north_dataframe['prefix'] = marci_north_dataframe['image'].apply(obs_prefix)
marci_north_dataframe['MY'] = marci_north_dataframe['prefix'].apply(lambda x: lookup_my(marci_date_conversion_csv, x))
marci_north_dataframe['Ls'] = marci_north_dataframe['image'].apply(lambda x: ls_mid_from_name(x))

# ==========================Manipulation==========================
moc_north_dataframe['avg_major_latitude'] = moc_north_dataframe[['major_end1_lat', 'major_end2_lat']].mean(axis=1)
marci_north_dataframe['avg_major_latitude'] = marci_north_dataframe[['major_end1_lat', 'major_end2_lat']].mean(axis=1)

moc_north_dataframe["Wrapped_Ls"] = wrap_ls(moc_north_dataframe["Ls"])
marci_north_dataframe["Wrapped_Ls"] = wrap_ls(marci_north_dataframe["Ls"])

# combined dataframe 
merged_df = pd.concat([
    moc_north_dataframe[['Wrapped_Ls', 'avg_major_latitude']],
    marci_north_dataframe[['Wrapped_Ls', 'avg_major_latitude']]
], ignore_index=True)

print(merged_df.head())
merged_ls, merged_lat = climatological_average(
    merged_df, bin_width=2,
    lo=merged_df["Wrapped_Ls"].min(), hi=merged_df["Wrapped_Ls"].max()
)

report_outside_window(moc_north_dataframe, "MOC north")
report_outside_window(marci_north_dataframe, "MARCI north")



# ==========================Plotting==========================
fig, ax = plt.subplots(figsize=(9, 5))

for my_val, group in moc_north_dataframe.groupby('MY'):
    sc = ax.scatter(group['Wrapped_Ls'], group['avg_major_latitude'],
                    label=f'MOC MY {my_val}', s=5)
    ls_avg, lat_avg = climatological_average(
        group, bin_width=MY_BIN_WIDTH,
        lo=LS_WINDOW_START, hi=LS_WINDOW_START + LS_WINDOW_SPAN)
    ax.plot(ls_avg, lat_avg, color=sc.get_facecolor()[0], linewidth=1)
for my_val, group in marci_north_dataframe.groupby('MY'):
    sc = ax.scatter(group['Wrapped_Ls'], group['avg_major_latitude'],
                    label=f'MARCI MY {my_val}', s=5)
    ls_avg, lat_avg = climatological_average(
        group, bin_width=MY_BIN_WIDTH,
        lo=LS_WINDOW_START, hi=LS_WINDOW_START + LS_WINDOW_SPAN)
    ax.plot(ls_avg, lat_avg, color=sc.get_facecolor()[0], linewidth=1)
ax.plot(merged_ls, merged_lat, color='black', label='Climatological Average', linewidth=1)
ax.set_xlim(LS_WINDOW_START, LS_WINDOW_START + LS_WINDOW_SPAN)
ax.set_xticks(np.arange(LS_WINDOW_START,
                        LS_WINDOW_START + LS_WINDOW_SPAN + 0.5,
                        LS_TICK_STEP))
ax.xaxis.set_major_formatter(FuncFormatter(ls_tick_label))
ax.set_xlabel('Ls (deg)')
ax.set_ylabel('Avg Major Latitude')
ax.legend(title='Mars Year')
ax.set_title('North polar cap - MOC vs MARCI')
plt.show()
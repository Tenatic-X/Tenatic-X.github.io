---
layout: project
date: 2026-07-12
category: personal
---

"""
S&P 500 Sector Heatmap Dashboard
---------------------------------
Run:
    pip install dash plotly pandas pyarrow
    python sp500_heatmap_dashboard.py
Open http://127.0.0.1:8050

Expects:
    data/parquet/sp500_close_prices.parquet
    data/parquet/sp500_open_prices.parquet
    data/parquet/sp500_volume.parquet
    data/sp500_sectors.csv   (Ticker, Sector, SubIndustry, Status, Name)
"""

import numpy as np
import pandas as pd
import yfinance as yf

import dash
from dash import dcc, html, dash_table, Input, Output, State, Patch, ctx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
CLOSE_PATH  = "data/parquet/sp500_close_prices.parquet"
OPEN_PATH   = "data/parquet/sp500_open_prices.parquet"
VOLUME_PATH = "data/parquet/sp500_volume.parquet"
SECTORS_CSV = "data/sp500_sectors.csv"

VIX_TICKER = "^VIX"
SPX_TICKER = "^GSPC"

METRIC_OPTIONS = [
    {"label": "Close-to-close change",                    "value": "close_change"},
    {"label": "Open-to-open change",                      "value": "open_change"},
    {"label": "Intraday performance (avg close/open − 1)", "value": "intraday"},
    {"label": "Volume change vs prior window",             "value": "volume_change"},
]

# Lookback windows for the Top/Worst performers section. "Prev. trading day"
# and "3 Trading Days" use actual prior rows in TRADING_DAYS (exact, not
# calendar-based); the week/month windows are calendar offsets back from the
# latest selected date, snapped to the nearest real trading day. "Wild Card"
# lets the person pick any arbitrary past date via a date picker.
TOP_WORST_WINDOWS = [
    {"label": "Prev. trading day", "value": "1d"},
    {"label": "3 Trading Days",    "value": "3d"},
    {"label": "1 Week",            "value": "1w"},
    {"label": "2 Week",            "value": "2w"},
    {"label": "3 Week",            "value": "3w"},
    {"label": "4 Week",            "value": "4w"},
    {"label": "2 Month",           "value": "2m"},
    {"label": "3 Month",           "value": "3m"},
    {"label": "🃏 Wild Card",       "value": "wildcard"},
]
TOP_WORST_OFFSETS = {
    "1w": pd.DateOffset(weeks=1),
    "2w": pd.DateOffset(weeks=2),
    "3w": pd.DateOffset(weeks=3),
    "4w": pd.DateOffset(weeks=4),
    "2m": pd.DateOffset(months=2),
    "3m": pd.DateOffset(months=3),
}
# Trading-day-index based windows (exact prior row, not a calendar offset)
TOP_WORST_TRADING_DAY_LOOKBACK = {"1d": 1, "3d": 3}

# ─────────────────────────────────────────────────────────────────────────────
# Dark-mode palette
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG      = "#0e1117"
DARK_SURFACE = "#161b22"
DARK_BORDER  = "#30363d"
DARK_TEXT    = "#e6edf3"
DARK_MUTED   = "#8b949e"
ACCENT_BLUE  = "#58a6ff"
ACCENT_RED   = "#f85149"

# Sector → distinct color (used for "all sub-industries" coloring)
SECTOR_COLORS = {
    # 12 perceptually distinct colors — matched across heatmap borders, perf lines, table, checklists
    "Communication Services":   "#4e9af1",   # sky blue
    "Consumer Discretionary":   "#f4a261",   # warm orange
    "Consumer Staples":         "#2ec4b6",   # cyan-teal  (was #2a9d8f — brightened)
    "Energy":                   "#ffd166",   # soft gold  (was #e9c46a)
    "Financials":               "#ef476f",   # hot pink-red (was #e76f51 coral — distinct from Energy)
    "Health Care":              "#06d6a0",   # mint green (was #57cc99)
    "Industrials":              "#a8dadc",   # powder blue-cyan
    "Information Technology":   "#c77dff",   # bright purple
    "Materials":                "#ff9f1c",   # amber  (was #b5838d mauve — too close to Financials)
    "Real Estate":              "#b5838d",   # mauve-rose  (was #6d6875 too dark)
    "Utilities":                "#f1c40f",   # bright yellow (distinct from Energy gold)
    "Delisted":                 "#8ecae6",   # pale steel blue (neutral, clearly separate)
}

# Translucent RGBA backing versions for text labels (sector color at ~40% opacity over dark bg)
def _hex_to_rgba(hex_color, alpha=0.35):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

SECTOR_LABEL_BG = {k: _hex_to_rgba(v, 0.38) for k, v in SECTOR_COLORS.items()}

DARK_READER_OVERRIDE_CSS = """
/* ── Dark Reader resistance — scoped to non-dropdown elements ────────────── */
/* We do NOT set darkreader-lock so Dark Reader CAN restyle the dropdowns.    */
/* Instead we lock only the structural UI and charts against color inversion. */

/* Lock body bg/text */
body, #react-entry-point, ._dash-loading {
    background-color: #0e1117 !important;
    color: #e6edf3 !important;
    forced-color-adjust: none !important;
}

/* Lock graph/chart containers and Dash layout wrappers — NOT Select/dropdown */
.dash-graph, .js-plotly-plot, .plotly, .plot-container,
.dash-table-container, .dash-spreadsheet-container,
.dash-spreadsheet, .dash-header,
.rc-tabs, .rc-tabs-tab,
.DateInput, .DateRangePicker, .SingleDatePicker,
.DateInput_input, .DateRangePickerInput,
.DayPicker, .CalendarMonth, .CalendarMonthGrid,
.CalendarDay, .DayPickerNavigation_button,
.dash-tab, ._dash-loading-callback-fragment {
    forced-color-adjust: none !important;
}

/* Exclude Dash Select / dropdown from forced-color-adjust
   so Dark Reader CAN restyle them (no rule applied = DR free to act) */
/* .Select, .Select-control, .Select-menu-outer → intentionally NOT locked */
"""

DARK_READER_META = (
    # Only set color-scheme: dark (not darkreader-lock) so Dark Reader
    # knows we're dark but doesn't fully skip the page — it will still
    # restyle elements that don't have forced-color-adjust: none,
    # which means our dropdowns get Dark Reader theming while charts stay fixed.
    '<meta name="color-scheme" content="dark">'
)

GRAPH_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

def load_price_data():
    close  = pd.read_parquet(CLOSE_PATH)
    open_  = pd.read_parquet(OPEN_PATH)
    volume = pd.read_parquet(VOLUME_PATH)
    for df in (close, open_, volume):
        df.index = pd.to_datetime(df.index).normalize()
        df.sort_index(inplace=True)
    return close, open_, volume


def load_sector_table():
    sectors = pd.read_csv(SECTORS_CSV)
    sectors["Sector"]      = sectors["Sector"].fillna("Delisted")
    sectors["SubIndustry"] = sectors["SubIndustry"].fillna("Delisted")
    return sectors.set_index("Ticker")


def _build_name_map(tbl):
    """Ticker → company name. Falls back to ticker if no Name column."""
    if "Name" in tbl.columns:
        return tbl["Name"].to_dict()
    return {}


def fetch_index_history(ticker, start="2015-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Load at startup
# ─────────────────────────────────────────────────────────────────────────────
print("Loading price data...")
df_close, df_open, df_volume = load_price_data()

print("Loading sector table...")
sector_table    = load_sector_table()
sector_map      = sector_table["Sector"].to_dict()
subindustry_map = sector_table["SubIndustry"].to_dict()
name_map        = _build_name_map(sector_table)

print("Loading SPX / VIX history...")
spx_df = fetch_index_history(SPX_TICKER, start=str(df_close.index.min().date()))
vix_df = fetch_index_history(VIX_TICKER, start=str(df_close.index.min().date()))

DATE_MIN     = df_close.index.min()
DATE_MAX     = df_close.index.max()
TRADING_DAYS = df_close.index

SECTOR_LIST = sorted(s for s in sector_table["Sector"].unique() if s != "Delisted")

# Delisted tickers carry Sector="Delisted" with no real GICS breakdown. We keep
# SECTOR_LIST as "real" GICS sectors (used wherever sub-industry drill-down
# logic assumes real sectors), but expose GROUP_LIST for anywhere we want
# Delisted to appear as a selectable/visible bucket alongside the real sectors.
HAS_DELISTED = (sector_table["Sector"] == "Delisted").any()
GROUP_LIST = SECTOR_LIST + (["Delisted"] if HAS_DELISTED else [])

SUBINDUSTRY_BY_SECTOR: dict[str, list[str]] = {}
for _sector in SECTOR_LIST:
    _tickers = [t for t, s in sector_map.items() if s == _sector]
    _subs = sorted(set(subindustry_map.get(t, "Unknown") for t in _tickers))
    SUBINDUSTRY_BY_SECTOR[_sector] = [s for s in _subs if s not in ("Unknown", "Delisted")]
if HAS_DELISTED:
    # Delisted has no real sub-industry breakdown -- treat it as one bucket
    # so it still works in drill-down / "all sub-industries" views.
    SUBINDUSTRY_BY_SECTOR["Delisted"] = ["Delisted"]

# sub-industry → sector lookup (for coloring)
SUBIND_TO_SECTOR: dict[str, str] = {}
for t, sub in subindustry_map.items():
    sec = sector_map.get(t, "Unknown")
    SUBIND_TO_SECTOR[sub] = sec


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def snap_to_trading_day(ts):
    ts = pd.Timestamp(ts).normalize()
    idx = TRADING_DAYS.searchsorted(ts, side="left")
    idx = min(idx, len(TRADING_DAYS) - 1)
    if idx > 0 and abs((TRADING_DAYS[idx] - ts).days) > abs((TRADING_DAYS[idx - 1] - ts).days):
        idx -= 1
    return TRADING_DAYS[idx]


def _extract_range_from_relayout(relayout):
    if not relayout:
        return None
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        return relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]
    if "xaxis.range" in relayout:
        rng = relayout["xaxis.range"]
        return rng[0], rng[1]
    return None


def _rescale_y_to_window(df, x_range):
    try:
        lo = pd.Timestamp(x_range[0]).normalize()
        hi = pd.Timestamp(x_range[1]).normalize()
    except Exception:
        return None
    visible = df.loc[lo:hi, "Close"]
    if visible.empty:
        return None
    ymin, ymax = visible.min(), visible.max()
    pad = (ymax - ymin) * 0.08 or ymax * 0.02
    return [ymin - pad, ymax + pad]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(start_date, end_date, metric, visible_sectors=None):
    start_date = snap_to_trading_day(pd.Timestamp(start_date))
    end_date   = snap_to_trading_day(pd.Timestamp(end_date))
    if end_date <= start_date:
        start_date, end_date = end_date, start_date

    window_close  = df_close.loc[start_date:end_date]
    window_open   = df_open.loc[start_date:end_date]
    window_volume = df_volume.loc[start_date:end_date]

    if window_close.shape[0] < 2:
        return pd.DataFrame(columns=["Ticker", "Sector", "SubIndustry", "Value", "AvgDollarVol"])

    c0, c1 = window_close.iloc[0], window_close.iloc[-1]
    o0, o1 = window_open.iloc[0],  window_open.iloc[-1]
    avg_dollar_vol = (window_volume * window_close).mean()

    if metric == "close_change":
        value = ((c1 / c0 - 1.0) * 100).round(2)
    elif metric == "open_change":
        value = ((o1 / o0 - 1.0) * 100).round(2)
    elif metric == "intraday":
        value = ((window_close / window_open - 1.0).mean() * 100).round(2)
    elif metric == "volume_change":
        n = window_volume.shape[0]
        prior_start = max(df_volume.index.searchsorted(start_date) - n, 0)
        prior_window = df_volume.iloc[prior_start:df_volume.index.searchsorted(start_date)]
        value = ((window_volume.mean() / prior_window.mean().replace(0, np.nan) - 1.0) * 100).round(2)
    else:
        raise ValueError(metric)

    out = pd.DataFrame({
        "Ticker":       value.index,
        "Value":        value.values,
        "AvgDollarVol": avg_dollar_vol.reindex(value.index).values,
        "LatestClose":  c1.reindex(value.index).values,
    })
    out["Sector"]      = out["Ticker"].map(sector_map).fillna("Unknown")
    out["SubIndustry"] = out["Ticker"].map(subindustry_map).fillna("Unknown")
    out["Name"]        = out["Ticker"].map(name_map).fillna(out["Ticker"])
    out.dropna(subset=["Value"], inplace=True)
    out["AvgDollarVol"] = out["AvgDollarVol"].fillna(out["AvgDollarVol"].median())
    out.loc[out["AvgDollarVol"] <= 0, "AvgDollarVol"] = out["AvgDollarVol"].median()

    if visible_sectors:
        out = out[out["Sector"].isin(visible_sectors)]

    out["Value"] = out["Value"].round(2)
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Performance data
# ─────────────────────────────────────────────────────────────────────────────

def compute_group_performance(start_date, end_date,
                               drill_sector=None,
                               all_subindustries=False,
                               all_subind_sectors=None):
    """
    drill_sector=None, all_subindustries=False
        → one line per GICS Sector + SPX
    drill_sector=<str>
        → sub-industries within that sector + Sector Avg + SPX
    all_subindustries=True
        → every sub-industry in every sector in all_subind_sectors,
          colored by sector (caller uses SUBIND_TO_SECTOR for colors)
    """
    start_date = snap_to_trading_day(pd.Timestamp(start_date))
    end_date   = snap_to_trading_day(pd.Timestamp(end_date))
    if end_date <= start_date:
        start_date, end_date = end_date, start_date

    window_close = df_close.loc[start_date:end_date]
    if window_close.shape[0] < 2:
        return pd.DataFrame()

    normalized = window_close / window_close.iloc[0] * 100.0

    # ── All sub-industries mode ──────────────────────────────────────────────
    if all_subindustries:
        sectors_to_show = all_subind_sectors or GROUP_LIST
        result = {}
        for sector in sectors_to_show:
            for subind in SUBINDUSTRY_BY_SECTOR.get(sector, []):
                cols = [t for t in normalized.columns
                        if sector_map.get(t) == sector and subindustry_map.get(t) == subind]
                if cols:
                    result[subind] = normalized[cols].mean(axis=1)
        perf_df = pd.DataFrame(result)
        spx_window = spx_df.loc[start_date:end_date, "Close"]
        if not spx_window.empty:
            perf_df["SPX"] = spx_window / spx_window.iloc[0] * 100.0
        return perf_df

    # ── Standard sector / drill mode ────────────────────────────────────────
    if drill_sector is None:
        group_of = sector_map
        groups   = GROUP_LIST
    else:
        tickers_in = [t for t, s in sector_map.items() if s == drill_sector]
        group_of   = {t: subindustry_map.get(t, "Unknown") for t in tickers_in}
        groups     = sorted(set(group_of.values()))

    result = {}
    for g in groups:
        cols = [t for t in normalized.columns if group_of.get(t) == g]
        if cols:
            result[g] = normalized[cols].mean(axis=1)

    if drill_sector is not None:
        sc = [t for t in normalized.columns if sector_map.get(t) == drill_sector]
        if sc:
            result[f"Sector Avg ({drill_sector})"] = normalized[sc].mean(axis=1)

    perf_df = pd.DataFrame(result)
    spx_window = spx_df.loc[start_date:end_date, "Close"]
    if not spx_window.empty:
        perf_df["SPX"] = spx_window / spx_window.iloc[0] * 100.0
    return perf_df


# ─────────────────────────────────────────────────────────────────────────────
# Top / worst performers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_top_worst_start(latest_date, window_value, wildcard_date=None):
    """Resolve the start date for a Top/Worst lookback window, anchored on
    `latest_date` (the heatmap's end date)."""
    latest_date = snap_to_trading_day(pd.Timestamp(latest_date))

    if window_value in TOP_WORST_TRADING_DAY_LOOKBACK:
        n = TOP_WORST_TRADING_DAY_LOOKBACK[window_value]
        idx = TRADING_DAYS.searchsorted(latest_date)
        prev_idx = max(idx - n, 0)
        return TRADING_DAYS[prev_idx]

    if window_value == "wildcard":
        if wildcard_date is None:
            # no date chosen yet -- fall back to 1 month ago so the chart/table
            # isn't empty before the person picks something
            return snap_to_trading_day(latest_date - pd.DateOffset(months=1))
        wc = snap_to_trading_day(pd.Timestamp(wildcard_date))
        return min(wc, latest_date)  # never let a future pick invert the window

    offset = TOP_WORST_OFFSETS.get(window_value)
    if offset is None:
        raise ValueError(window_value)
    return snap_to_trading_day(latest_date - offset)


def _pct_change_for_window(tickers, start_date, end_date):
    """Close-to-close % change for a fixed list of tickers over [start_date, end_date]."""
    cols = [t for t in tickers if t in df_close.columns]
    window_close = df_close.loc[start_date:end_date, cols]
    if window_close.shape[0] < 2:
        return pd.Series(np.nan, index=tickers)
    c0, c1 = window_close.iloc[0], window_close.iloc[-1]
    pct = ((c1 / c0 - 1.0) * 100.0)
    return pct.reindex(tickers)


def compute_top_worst_performance(latest_date, window_value, wildcard_date=None, top_n=20):
    """Close-to-close % change for every ticker over the chosen window,
    ending at `latest_date`. Returns (top_df, worst_df, start_date, end_date),
    each df sorted with the most extreme mover first."""
    end_date   = snap_to_trading_day(pd.Timestamp(latest_date))
    start_date = resolve_top_worst_start(end_date, window_value, wildcard_date)
    if start_date >= end_date:
        idx = TRADING_DAYS.searchsorted(end_date)
        start_date = TRADING_DAYS[max(idx - 1, 0)]

    window_close = df_close.loc[start_date:end_date]
    if window_close.shape[0] < 2:
        empty = pd.DataFrame(columns=["Ticker", "Name", "Sector", "PctChange"])
        return empty, empty, start_date, end_date

    c0, c1 = window_close.iloc[0], window_close.iloc[-1]
    pct = ((c1 / c0 - 1.0) * 100.0)

    out = pd.DataFrame({"Ticker": pct.index, "PctChange": pct.values})
    out.dropna(subset=["PctChange"], inplace=True)
    out["Sector"] = out["Ticker"].map(sector_map).fillna("Unknown")
    out["Name"]   = out["Ticker"].map(name_map).fillna(out["Ticker"])

    top_df   = out.nlargest(top_n, "PctChange").sort_values("PctChange", ascending=False)
    worst_df = out.nsmallest(top_n, "PctChange").sort_values("PctChange", ascending=True)
    return top_df, worst_df, start_date, end_date


def compute_multi_window_breakdown(tickers, latest_date, wildcard_date=None):
    """For a fixed list of tickers, compute % change for every Top/Worst
    window (3d, 1w, 2w, ... ) anchored at `latest_date`, plus the Wild Card
    window if a date has been picked. Returns a DataFrame indexed by Ticker,
    one column per window value (column id == window value, e.g. '1w')."""
    end_date = snap_to_trading_day(pd.Timestamp(latest_date))
    cols = {}
    for opt in TOP_WORST_WINDOWS:
        wv = opt["value"]
        if wv == "wildcard" and wildcard_date is None:
            continue  # no Wild Card column until a date is actually picked
        start_date = resolve_top_worst_start(end_date, wv, wildcard_date)
        if start_date >= end_date:
            idx = TRADING_DAYS.searchsorted(end_date)
            start_date = TRADING_DAYS[max(idx - 1, 0)]
        cols[wv] = _pct_change_for_window(tickers, start_date, end_date)
    table = pd.DataFrame(cols, index=tickers)
    return table


def build_top_worst_table_records(top_df, worst_df, latest_date, wildcard_date=None):
    """Builds the row dicts + column defs for the Top/Worst breakdown tables.
    Each row has Ticker/Name/Sector, a display string per window (e.g. '5.23%'
    in column id '1w'), and a parallel '<id>__raw' numeric field used only for
    conditional color styling (not rendered as its own column)."""
    all_tickers = list(top_df["Ticker"]) + list(worst_df["Ticker"])
    breakdown = compute_multi_window_breakdown(all_tickers, latest_date, wildcard_date)

    active_window_values = [c for c in breakdown.columns]
    columns = [{"name": "Ticker", "id": "Ticker"},
               {"name": "Name", "id": "Name"},
               {"name": "Sector", "id": "Sector"}]
    label_map = {opt["value"]: opt["label"] for opt in TOP_WORST_WINDOWS}
    for wv in active_window_values:
        name = label_map.get(wv, wv)
        if wv == "wildcard" and wildcard_date is not None:
            name = f"🃏 {pd.Timestamp(wildcard_date).date()}"
        columns.append({"name": name, "id": wv})

    def _rows(df):
        records = []
        for _, row in df.iterrows():
            tk = row["Ticker"]
            rec = {"Ticker": tk, "Name": row["Name"], "Sector": row["Sector"],
                   "_sector_color": SECTOR_COLORS.get(row["Sector"], DARK_TEXT)}
            for wv in active_window_values:
                val = breakdown.loc[tk, wv] if tk in breakdown.index else np.nan
                rec[wv] = "—" if pd.isna(val) else f"{val:+.2f}%"
                rec[f"{wv}__raw"] = None if pd.isna(val) else float(val)
            records.append(rec)
        return records

    return _rows(top_df), _rows(worst_df), columns, active_window_values


TABLE_DARK_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_header={
        "backgroundColor": DARK_SURFACE, "color": DARK_TEXT,
        "border": f"1px solid {DARK_BORDER}", "fontWeight": "bold",
    },
    style_cell={
        "backgroundColor": DARK_BG, "color": "#111111",
        "border": f"1px solid {DARK_BORDER}", "fontSize": "12px",
        "padding": "6px 10px", "textAlign": "center",
        "fontFamily": "Inter, Arial, sans-serif",
        "userSelect": "text",        # allow text highlighting for copy-paste
    },
    style_cell_conditional=[
        {"if": {"column_id": "Name"},   "textAlign": "left", "minWidth": "150px"},
        {"if": {"column_id": "Sector"}, "textAlign": "left", "minWidth": "150px"},
        {"if": {"column_id": "Ticker"}, "fontWeight": "bold"},
    ],
    # Disable row/cell selection so no bounding box appears on click
    row_selectable=False,
    cell_selectable=False,
)


def _pct_to_color(val, vmin, vmax):
    """Map a value to a hex color using the finviz red→grey→green scale,
    clamped within [vmin, vmax] and respecting sign (red only for negatives,
    green only for positives)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if val >= 0:
        if vmax <= 0:
            return "#383838"
        t = min(val / vmax, 1.0)
        r = int(0x38 + t * (0x00 - 0x38))
        g = int(0x38 + t * (0xb3 - 0x38))
        b = int(0x38 + t * (0x00 - 0x38))
    else:
        if vmin >= 0:
            return "#383838"
        t = min(abs(val) / abs(vmin), 1.0)
        r = int(0x38 + t * (0xff - 0x38))
        g = int(0x38 + t * (0x2e - 0x38))
        b = int(0x38 + t * (0x2e - 0x38))
    return f"#{r:02x}{g:02x}{b:02x}"


def make_window_table_style(active_window_values, rank_window_value, top_rows, worst_rows):
    """
    Build style_data_conditional with per-window normalized background colors.
    Green/red intensity reflects magnitude within each window's own distribution.
    Red is only applied to negative values; green only to positive values.
    The active ranking window column gets a blue border highlight.
    Ticker and Sector text is colored by sector.
    """
    all_rows = top_rows + worst_rows
    style = []

    # ── Sector-colored border + translucent bg on Ticker, Name & Sector cells; white text ──
    seen_sectors = set()
    for row in all_rows:
        sector = row.get("Sector", "")
        border_color = row.get("_sector_color", DARK_BORDER)
        if sector and sector not in seen_sectors:
            seen_sectors.add(sector)
            bg = SECTOR_LABEL_BG.get(sector, "rgba(40,40,40,0.18)")
            style.append({
                "if": {"filter_query": f'{{Sector}} = "{sector}"', "column_id": "Ticker"},
                "color": "#ffffff",
                "backgroundColor": bg,
                "borderLeft": f"3px solid {border_color}",
                "fontWeight": "bold",
            })
            style.append({
                "if": {"filter_query": f'{{Sector}} = "{sector}"', "column_id": "Name"},
                "color": "#ffffff",
                "backgroundColor": bg,
                "borderLeft": f"3px solid {border_color}",
            })
            style.append({
                "if": {"filter_query": f'{{Sector}} = "{sector}"', "column_id": "Sector"},
                "color": "#ffffff",
                "backgroundColor": bg,
                "borderLeft": f"3px solid {border_color}",
            })

    for wv in active_window_values:
        raw_vals = [r.get(f"{wv}__raw") for r in all_rows
                    if r.get(f"{wv}__raw") is not None]
        positives = [v for v in raw_vals if v > 0]
        negatives = [v for v in raw_vals if v < 0]
        vmax = max(positives) if positives else 0.0
        vmin = min(negatives) if negatives else 0.0

        # Build a color rule for each distinct raw value
        seen = set()
        for row in all_rows:
            raw = row.get(f"{wv}__raw")
            if raw is None or raw in seen:
                continue
            seen.add(raw)
            color = _pct_to_color(raw, vmin, vmax)
            if color:
                style.append({
                    "if": {"filter_query": f"{{{wv}__raw}} = {raw}", "column_id": wv},
                    "backgroundColor": color,
                    "color": "#ffffff",
                })

        if wv == rank_window_value:
            style.append({
                "if": {"column_id": wv},
                "border": f"1px solid {ACCENT_BLUE}",
                "fontWeight": "bold",
            })

    # On hover/active, flip ALL columns to a dark background with dark text
    # so Dash's unavoidable row highlight just looks like a subtle darkening.
    for col in ("Ticker", "Name", "Sector"):
        style.append({
            "if": {"state": "active", "column_id": col},
            "backgroundColor": "#0a0d12",
            "color": "#0a0d12",
            "border": f"1px solid {DARK_BORDER}",
        })

    return style


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

PERF_HEIGHT = int(520 * 1.7 * 0.80)   # 1.7× original, then −20%

_AXIS = dict(gridcolor=DARK_BORDER, linecolor=DARK_BORDER, tickformat="%b %d<br>%Y")
_YAXIS = dict(gridcolor=DARK_BORDER, linecolor=DARK_BORDER,
              title="Normalized performance (start = 100)")
_LEGEND = dict(orientation="h", yanchor="bottom", y=1.01, x=0,
               bgcolor=DARK_SURFACE, bordercolor=DARK_BORDER)
_LAYOUT_BASE = dict(
    margin=dict(l=40, r=20, t=50, b=10),
    paper_bgcolor=DARK_SURFACE, plot_bgcolor=DARK_SURFACE,
    font=dict(color=DARK_TEXT),
    xaxis=_AXIS, yaxis=_YAXIS, legend=_LEGEND,
)


def make_index_figure(df, title, color, uid):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines",
        line=dict(color=color, width=1.4), name=title,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=DARK_TEXT)),
        height=260,
        margin=dict(l=40, r=20, t=40, b=10),
        xaxis=dict(type="date", tickformat="%b %d<br>%Y",
                   gridcolor=DARK_BORDER, linecolor=DARK_BORDER),
        yaxis=dict(gridcolor=DARK_BORDER, linecolor=DARK_BORDER),
        dragmode="zoom", showlegend=False,
        paper_bgcolor=DARK_SURFACE, plot_bgcolor=DARK_SURFACE,
        font=dict(color=DARK_TEXT),
        uirevision=uid,
    )
    return fig


# Finviz-style colorscale: red (loss) → dark grey (flat) → green (gain)
FINVIZ_COLORSCALE = [
    [0.0,  "#ff2e2e"],
    [0.5,  "#383838"],
    [1.0,  "#00b300"],
]

def make_treemap(metrics_df, metric_label):
    if metrics_df.empty:
        return go.Figure().update_layout(
            title="No data for this window (need at least 2 trading days)",
            paper_bgcolor=DARK_SURFACE, font=dict(color=DARK_TEXT),
        )
    df = metrics_df.copy()
    bound = max(abs(df["Value"].quantile(0.02)), abs(df["Value"].quantile(0.98)), 0.5)

    fig = px.treemap(
        df,
        path=["Sector", "SubIndustry", "Ticker"],
        values="AvgDollarVol",
        color="Value",
        color_continuous_scale=FINVIZ_COLORSCALE,
        range_color=[-bound, bound],
        hover_data=["Value", "Name", "LatestClose", "Sector", "SubIndustry"]
    )

    fig.update_traces(
        # Plain white text — ticker bold + % change, no sector coloring in heatmap cells
        texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}%",
        textposition="middle center",
        textfont=dict(color="white", size=11),
        insidetextfont=dict(color="white", size=11),

        tiling=dict(pad=1),

        marker=dict(
            line=dict(color="#000000", width=0.5),
            pad=dict(t=18, l=1, r=1, b=1),
        ),

        hoverlabel=dict(
            bgcolor="#1c2128",
            bordercolor=DARK_BORDER,
            font=dict(color="#ffffff", size=12),
        ),
        hovertemplate=(
            "<b>%{customdata[1]}</b> (%{label})<br>"
            "Sub-industry: %{customdata[4]}<br>"
            "Price: $%{customdata[2]:.2f}<br>"
            "Change: <b>%{customdata[0]:.2f}%</b>"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        title=dict(text=f"S&P 500 Sector Heatmap — {metric_label}", font=dict(color=DARK_TEXT)),
        margin=dict(l=10, r=10, t=50, b=50),
        height=850,
        paper_bgcolor=DARK_SURFACE,
        font=dict(color=DARK_TEXT),
        coloraxis_colorbar=dict(
            title=dict(text="%", font=dict(color=DARK_TEXT)),
            ticksuffix="%", tickformat=".2f", tickfont=dict(color=DARK_TEXT),
        ),
    )
    return fig


def make_performance_figure(perf_df, drill_sector):
    if perf_df.empty:
        return go.Figure().update_layout(
            title="No data for this window (need at least 2 trading days)",
            paper_bgcolor=DARK_SURFACE, font=dict(color=DARK_TEXT),
        )
    fig = go.Figure()
    for col in perf_df.columns:
        is_spx = col == "SPX"
        is_avg = "Sector Avg" in col
        fig.add_trace(go.Scatter(
            x=perf_df.index, y=perf_df[col], mode="lines", name=col,
            line=dict(
                width=3 if is_spx else (2.5 if is_avg else 1.6),
                dash="dash" if is_spx else ("dot" if is_avg else "solid"),
                color="white" if is_spx else (
                    SECTOR_COLORS.get(drill_sector) if is_avg else None
                ),
            ),
        ))
    title_txt = ("Sector performance vs SPX" if drill_sector is None
                 else f"{drill_sector} — sub-industries vs SPX")
    fig.update_layout(
        title=dict(text=title_txt, font=dict(color=DARK_TEXT)),
        height=PERF_HEIGHT,
        uirevision="perf-figure",
        **_LAYOUT_BASE,
    )
    return fig


def make_all_subind_figure(perf_df, sectors_shown, hidden_subindustries=None):
    """Every sub-industry in one chart, lines colored by their sector.
    Sub-industries in hidden_subindustries are rendered with visible='legendonly'
    so they disappear from the chart but can still be toggled back via the legend."""
    hidden_set = set(hidden_subindustries or [])
    if perf_df.empty:
        return go.Figure().update_layout(
            title="Select at least one sector below",
            paper_bgcolor=DARK_SURFACE, font=dict(color=DARK_TEXT),
        )
    fig = go.Figure()
    # SPX last so it's always on top
    cols_no_spx = [c for c in perf_df.columns if c != "SPX"]
    for col in cols_no_spx:
        sector = SUBIND_TO_SECTOR.get(col, "Unknown")
        color  = SECTOR_COLORS.get(sector, "#aaaaaa")
        fig.add_trace(go.Scatter(
            x=perf_df.index, y=perf_df[col], mode="lines",
            name=col,
            customdata=[[col]] * len(perf_df),
            visible="legendonly" if col in hidden_set else True,
            legendgroup=sector,
            legendgrouptitle=dict(text=sector) if col == cols_no_spx[
                next((i for i, c in enumerate(cols_no_spx)
                      if SUBIND_TO_SECTOR.get(c) == sector), 0)
            ] else None,
            line=dict(color=color, width=1.4),
            hovertemplate=f'<b>{col}</b><br>({sector})<br>Date: %{{x|%b %d, %Y}}<br>%{{y:.2f}}<br><i>Click line to hide</i><extra></extra>',
        ))
    if "SPX" in perf_df.columns:
        fig.add_trace(go.Scatter(
            x=perf_df.index, y=perf_df["SPX"], mode="lines", name="SPX",
            line=dict(color="white", width=2.5, dash="dash"),
        ))

    fig.update_layout(
        title=dict(
            font=dict(color=DARK_TEXT),
        ),
        height=PERF_HEIGHT,
        uirevision="all-subind-figure",
        **_LAYOUT_BASE,
    )
    # Override legend separately to avoid duplicate keyword with _LAYOUT_BASE
    fig.update_layout(
        legend=dict(
            orientation="v", x=1.02, y=1.0, xanchor="left", yanchor="top",
            bgcolor=DARK_SURFACE, bordercolor=DARK_BORDER,
            groupclick="toggleitem",
            font=dict(size=1),
        ),
    )
    return fig


TOP_WORST_HEIGHT = 620

def make_top_worst_figure(top_df, worst_df, window_label, start_date, end_date):
    if top_df.empty and worst_df.empty:
        return go.Figure().update_layout(
            title="No data for this window (need at least 2 trading days)",
            paper_bgcolor=DARK_SURFACE, font=dict(color=DARK_TEXT),
        )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Top {len(top_df)} — {window_label}",
                         f"Worst {len(worst_df)} — {window_label}"),
        horizontal_spacing=0.14,
    )

    def _hover(df):
        return [
            f"<b>{n}</b> ({t})<br>Sector: {s}<br>Change: <b>{v:.2f}%</b>"
            for t, n, s, v in zip(df["Ticker"], df["Name"], df["Sector"], df["PctChange"])
        ]

    def _bar_colors(df, is_top):
        """Per-bar color: sector color blended toward green (top) or red (worst)."""
        return [SECTOR_COLORS.get(s, "#aaaaaa") for s in df["Sector"]]

    fig.add_trace(
        go.Bar(
            x=top_df["PctChange"], y=top_df["Ticker"], orientation="h",
            marker_color=_bar_colors(top_df, True),
            marker_line=dict(color="#00b300", width=1),
            text=top_df["PctChange"].map(lambda v: f"{v:.2f}%"),
            textposition="outside", textfont=dict(color=DARK_TEXT, size=10),
            hovertext=_hover(top_df), hovertemplate="%{hovertext}<extra></extra>", name="Top",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=worst_df["PctChange"], y=worst_df["Ticker"], orientation="h",
            marker_color=_bar_colors(worst_df, False),
            marker_line=dict(color="#ff2e2e", width=1),
            text=worst_df["PctChange"].map(lambda v: f"{v:.2f}%"),
            textposition="outside", textfont=dict(color=DARK_TEXT, size=10),
            hovertext=_hover(worst_df), hovertemplate="%{hovertext}<extra></extra>", name="Worst",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(gridcolor=DARK_BORDER, linecolor=DARK_BORDER, ticksuffix="%",
                      zerolinecolor=DARK_BORDER)
    fig.update_yaxes(gridcolor=DARK_BORDER, linecolor=DARK_BORDER, automargin=True)
    fig.update_annotations(font=dict(color=DARK_TEXT))

    fig.update_layout(
        showlegend=False,
        height=TOP_WORST_HEIGHT,
        margin=dict(l=10, r=20, t=50, b=10),
        paper_bgcolor=DARK_SURFACE, plot_bgcolor=DARK_SURFACE,
        font=dict(color=DARK_TEXT),
        title=dict(
            text=f"Best/worst movers  |  {start_date.date()} → {end_date.date()}",
            font=dict(color=DARK_TEXT),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dropdown CSS injection (forces dark bg on Dash Select menus)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Dropdown CSS injection (forces dark bg on Dash Select menus)
# ─────────────────────────────────────────────────────────────────────────────
DROPDOWN_CSS = """
/* Fix for new Dash components and the primary dropdown outer shell */
.dash-dropdown .Select-control,
.Select-control,
.Select-multi-value-wrapper,
.dropdown-container,
div[data-dash-is-loading] .dropdown {
    background-color: #0e1117 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}

/* Force text inside the input box to be visible and clear */
.Select-value-label,
.Select-placeholder,
.Select-input input,
.Select-value {
    color: #e6edf3 !important;
}

/* The actual menu list that drops down when clicked */
.Select-menu-outer,
.Select-menu {
    background-color: #0e1117 !important;
    border: 1px solid #30363d !important;
}

/* Individual options within the dropdown menu */
.VirtualizedSelectOption,
.Select-option {
    background-color: #0e1117 !important;
    color: #e6edf3 !important;
}

/* When you hover over an item in the menu list */
.Select-option.is-focused,
.VirtualizedSelectFocusedOption,
.Select-option:hover {
    background-color: #30363d !important;
    color: #ffffff !important;
}

/* When an item is actively selected inside the dropdown menu */
.Select-option.is-selected {
    background-color: #1f6feb !important;
    color: #ffffff !important;
}

/* The small arrow icon on the right side of the box */
.Select-arrow { 
    border-top-color: #8b949e !important; 
}
.Select-arrow-zone:hover .Select-arrow {
    border-top-color: #e6edf3 !important;
}

/* ── DatePickerRange / DatePickerSingle ── */
.DateInput_input, .DateInput, .DateRangePickerInput,
.SingleDatePickerInput, .DateRangePicker_picker,
.DayPicker, .DayPicker__withBorder,
.DayPickerNavigation_button, .CalendarMonth,
.CalendarMonthGrid, .CalendarDay__default {
    background-color: #0e1117 !important;
    color: #e6edf3 !important;
    border-color: #30363d !important;
}
.DateInput_input {
    font-size: 13px !important;
    padding: 4px 8px !important;
}
.DateRangePickerInput {
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}
.DateInput_input__focused {
    border-bottom: 2px solid #58a6ff !important;
}
.CalendarDay__default {
    background: #0e1117 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
}
.CalendarDay__default:hover {
    background: #30363d !important;
    color: #ffffff !important;
}
.CalendarDay__selected, .CalendarDay__selected:hover {
    background: #1f6feb !important;
    color: #ffffff !important;
    border-color: #1f6feb !important;
}
.CalendarDay__selected_span {
    background: #163b6e !important;
    color: #e6edf3 !important;
    border-color: #30363d !important;
}
.CalendarDay__hovered_span, .CalendarDay__hovered_span:hover {
    background: #1e3a5f !important;
    color: #e6edf3 !important;
}
.DayPickerNavigation_button__default {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}
.DayPickerNavigation_svg__horizontal { fill: #e6edf3 !important; }
.CalendarMonth_caption { color: #e6edf3 !important; }
.DayPicker_weekHeader_li small { color: #8b949e !important; }
.DateRangePickerInput_arrow_svg { fill: #8b949e !important; }

/* ── DataTable: kill selection bounding box, keep text selectable ── */
.dash-spreadsheet td.focused,
.dash-spreadsheet td.cell--selected,
.dash-spreadsheet-container td:focus,
.dash-spreadsheet td:focus {
    outline: none !important;
    box-shadow: none !important;
    border-color: #30363d !important;
}
.dash-spreadsheet .dash-cell-value {
    user-select: text !important;
    cursor: text !important;
}
.dash-spreadsheet td {
    cursor: default !important;
}

/* ── Dark Hover Effect ONLY for Ticker, Name, and Sector Columns ── */
#top-worst-table-top tr:hover td[data-dash-column="Ticker"],
#top-worst-table-top tr:hover td[data-dash-column="Name"],
#top-worst-table-top tr:hover td[data-dash-column="Sector"],
#top-worst-table-worst tr:hover td[data-dash-column="Ticker"],
#top-worst-table-worst tr:hover td[data-dash-column="Name"],
#top-worst-table-worst tr:hover td[data-dash-column="Sector"] {
    /* Only overlays a dark tint on these specific text columns */
    box-shadow: inset 0 0 0 9999px rgba(0, 0, 0, 0.55) !important;
}

#top-worst-table-top tr:hover td[data-dash-column="Ticker"] .dash-cell-value,
#top-worst-table-top tr:hover td[data-dash-column="Name"] .dash-cell-value,
#top-worst-table-top tr:hover td[data-dash-column="Sector"] .dash-cell-value,
#top-worst-table-worst tr:hover td[data-dash-column="Ticker"] .dash-cell-value,
#top-worst-table-worst tr:hover td[data-dash-column="Name"] .dash-cell-value,
#top-worst-table-worst tr:hover td[data-dash-column="Sector"] .dash-cell-value {
    background-color: transparent !important;
    color: #ffffff !important;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

# Inject CSS into <head> — Dark Reader meta tags + our override CSS + dropdown styles
app.index_string = app.index_string.replace(
    "<head>",
    f"<head>{DARK_READER_META}",
).replace(
    "</head>",
    f"<style>{DARK_READER_OVERRIDE_CSS}{DROPDOWN_CSS}</style></head>",
)

drill_options = (
    [{"label": "All sectors", "value": "ALL"},
     {"label": "── All sub-industries (combined view) ──", "value": "ALL_SUBIND"}]
    + [{"label": s, "value": s} for s in GROUP_LIST]
)

def _sector_label(s):
    """Colored sector label for checklists."""
    color = SECTOR_COLORS.get(s, DARK_TEXT)
    return html.Span(
        f"  {s}",
        style={"color": color, "fontWeight": "500"},
    )

CHECKLIST_STYLE = dict(
    display="flex", flexDirection="column", gap="4px",
    maxHeight="340px", overflowY="auto",
    padding="8px 12px", border=f"1px solid {DARK_BORDER}",
    borderRadius="6px", backgroundColor=DARK_BG,
)
LABEL_STYLE = dict(color=DARK_MUTED, fontSize="12px", marginBottom="4px")

DD_STYLE = dict(
    backgroundColor=DARK_BG, color=DARK_TEXT,
    border=f"1px solid {DARK_BORDER}",
)

app.layout = html.Div(
    style={
        "fontFamily": "Inter, Arial, sans-serif",
        "padding": "16px",
        "backgroundColor": DARK_BG,
        "color": DARK_TEXT,
        "minHeight": "100vh",
    },
    children=[
        html.H2("S&P 500 Performance Dashboard", style={"color": DARK_TEXT, "marginTop": 0}),
        html.P(
            "Scroll to zoom SPX/VIX — snaps to nearest trading day. "
            "Click 'Apply selection' to recompute all charts.",
            style={"color": DARK_MUTED},
        ),

        # ── Index charts ──────────────────────────────────────────────────
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            dcc.Graph(id="spx-graph",
                      figure=make_index_figure(spx_df, "S&P 500 (^GSPC)", ACCENT_BLUE, "spx"),
                      config=GRAPH_CONFIG, style={"flex": 1}),
            dcc.Graph(id="vix-graph",
                      figure=make_index_figure(vix_df, "VIX (^VIX)", ACCENT_RED, "vix"),
                      config=GRAPH_CONFIG, style={"flex": 1}),
        ]),

        # ── Controls ─────────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "alignItems": "flex-start",
                   "gap": "20px", "margin": "16px 0", "flexWrap": "wrap"},
            children=[
                html.Div([
                    html.Label("Selected window (editable):", style=LABEL_STYLE),
                    dcc.DatePickerRange(
                        id="date-range",
                        min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX,
                        start_date=DATE_MAX - pd.Timedelta(days=5),
                        end_date=DATE_MAX,
                    ),
                ]),
                html.Div([
                    html.Label("Heatmap metric:", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="metric-dropdown",
                        options=METRIC_OPTIONS, value="close_change", clearable=False,
                        style={"width": "340px", **DD_STYLE},
                    ),
                ]),
                html.Div([
                    html.Br(),
                    html.Button("Apply selection", id="apply-button", n_clicks=0,
                                style={
                                    "padding": "8px 20px", "fontWeight": "bold",
                                    "backgroundColor": ACCENT_BLUE, "color": "#fff",
                                    "border": "none", "borderRadius": "6px", "cursor": "pointer",
                                }),
                ]),
            ],
        ),

        # ── Heatmap sector visibility ─────────────────────────────────────
        html.Details([
            html.Summary(
                "▸ Sector visibility in heatmap (click to expand)",
                style={"cursor": "pointer", "color": DARK_MUTED,
                       "fontSize": "13px", "marginBottom": "8px"},
            ),
            dcc.Checklist(
                id="sector-visibility",
                options=[{"label": _sector_label(s), "value": s} for s in GROUP_LIST],
                value=GROUP_LIST,
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "flex", "alignItems": "center",
                            "fontSize": "13px", "padding": "2px 0"},
                style=CHECKLIST_STYLE,
            ),
        ], style={"marginBottom": "12px"}),

        # ── Heatmap ───────────────────────────────────────────────────────
        dcc.Graph(id="heatmap-graph"),

        html.Hr(style={"margin": "30px 0", "borderColor": DARK_BORDER}),

        # ── Performance section ───────────────────────────────────────────
        html.H3("Sector / sub-industry performance vs SPX",
                style={"color": DARK_TEXT, "marginBottom": "6px"}),
        html.P(
            "'All sectors' → one line per sector. "
            "'All sub-industries' → every sub-industry in one chart, colored by sector — "
            "use the sector filter below to narrow it down. "
            "Pick a specific sector to drill into its sub-industries. "
            "Clicking a sector block in the heatmap auto-selects it here. "
            "In the All sub-industries view, click any line to hide it (use Restore to bring back).",
            style={"color": DARK_MUTED, "fontSize": "13px"},
        ),

        # Drill dropdown — wide, no scroll needed
        dcc.Dropdown(
            id="drill-dropdown",
            options=drill_options, value="ALL", clearable=False,
            style={"width": "100%", "maxWidth": "860px",
                   "marginBottom": "10px", **DD_STYLE},
        ),

        # Sector filter — only shown when ALL_SUBIND is selected
        # Rendered as a collapsible <details> so it never steals graph vertical space.
        # The wrapper div is toggled visible/hidden by the callback, but because it
        # uses <details> the content is collapsed by default and takes only ~24 px.
        # Stores the set of sub-industries the user has double-clicked away
        dcc.Store(id="hidden-subindustries", data=[]),

        html.Div(
            id="all-subind-filter-wrapper",
            style={"display": "none", "marginBottom": "6px"},
            children=[
                html.Details([
                    html.Summary(
                        "▸ Filter sectors in this view (click to expand)",
                        style={"cursor": "pointer", "color": DARK_MUTED,
                               "fontSize": "12px"},
                    ),
                    dcc.Checklist(
                        id="all-subind-sector-filter",
                        options=[{"label": _sector_label(s), "value": s} for s in GROUP_LIST],
                        value=GROUP_LIST,
                        inputStyle={"marginRight": "5px"},
                        labelStyle={"display": "inline-flex", "alignItems": "center",
                                    "fontSize": "12px",
                                    "marginRight": "14px", "padding": "2px 0"},
                        style={
                            "display": "flex", "flexWrap": "wrap", "gap": "2px",
                            "padding": "8px 12px", "marginTop": "6px",
                            "border": f"1px solid {DARK_BORDER}",
                            "borderRadius": "6px", "backgroundColor": DARK_BG,
                        },
                    ),
                ]),
                # Reset button + hidden-count label — sits below the sector checklist
                html.Div(
                    style={"display": "flex", "alignItems": "center",
                           "gap": "12px", "marginTop": "8px"},
                    children=[
                        html.Button(
                            "↺ Restore hidden sub-industries",
                            id="reset-hidden-subind-btn",
                            n_clicks=0,
                            style={
                                "padding": "5px 14px", "fontSize": "12px",
                                "backgroundColor": DARK_SURFACE,
                                "color": DARK_TEXT,
                                "border": f"1px solid {DARK_BORDER}",
                                "borderRadius": "6px", "cursor": "pointer",
                            },
                        ),
                        html.Span(
                            id="hidden-subind-count",
                            style={"color": DARK_MUTED, "fontSize": "12px"},
                        ),
                    ],
                ),
            ],
        ),

        dcc.Graph(id="performance-graph"),

        html.Hr(style={"margin": "30px 0", "borderColor": DARK_BORDER}),

        # ── Top / Worst performers section ──────────────────────────────────
        html.H3("Top 20 / Worst 20 performers",
                style={"color": DARK_TEXT, "marginBottom": "6px"}),
        html.P(
            "Anchored on the latest date selected for the heatmap (the date-range's "
            "end date). Pick a lookback window below to see the best/worst 20 "
            "tickers from that many trading days/weeks/months ago up to that date.",
            style={"color": DARK_MUTED, "fontSize": "13px"},
        ),

        dcc.Tabs(
            id="top-worst-window",
            value="1d",
            children=[
                dcc.Tab(
                    label=opt["label"], value=opt["value"],
                    style={
                        "backgroundColor": DARK_BG, "color": DARK_MUTED,
                        "border": f"1px solid {DARK_BORDER}", "padding": "8px 4px",
                    },
                    selected_style={
                        "backgroundColor": ACCENT_BLUE, "color": "#ffffff",
                        "border": f"1px solid {ACCENT_BLUE}", "padding": "8px 4px",
                        "fontWeight": "bold",
                    },
                )
                for opt in TOP_WORST_WINDOWS
            ],
            style={"marginBottom": "10px", "maxWidth": "860px"},
        ),

        # Wild Card date picker -- only shown when that tab is active
        html.Div(
            id="wildcard-date-wrapper",
            style={"display": "none", "marginBottom": "10px"},
            children=[
                html.Label("Wild Card date (compare latest date back to this date):",
                           style=LABEL_STYLE),
                dcc.DatePickerSingle(
                    id="wildcard-date-picker",
                    min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX,
                    date=DATE_MAX - pd.Timedelta(days=30),
                ),
            ],
        ),

        html.H4("Top 20 movers — full period breakdown",
                style={"color": DARK_TEXT, "marginTop": "20px", "marginBottom": "6px"}),
        dash_table.DataTable(id="top-worst-table-top", **TABLE_DARK_STYLE),

        html.H4("Worst 20 movers — full period breakdown",
                style={"color": DARK_TEXT, "marginTop": "20px", "marginBottom": "6px"}),
        dash_table.DataTable(id="top-worst-table-worst", **TABLE_DARK_STYLE),

    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("spx-graph", "figure", allow_duplicate=True),
    Input("spx-graph", "relayoutData"),
    prevent_initial_call=True,
)
def rescale_spx_y(relayout):
    rng = _extract_range_from_relayout(relayout)
    if rng is None: raise dash.exceptions.PreventUpdate
    y = _rescale_y_to_window(spx_df, rng)
    if y is None: raise dash.exceptions.PreventUpdate
    p = Patch()
    p["layout"]["yaxis"]["range"]     = y
    p["layout"]["yaxis"]["autorange"] = False
    return p


@app.callback(
    Output("vix-graph", "figure", allow_duplicate=True),
    Input("vix-graph", "relayoutData"),
    prevent_initial_call=True,
)
def rescale_vix_y(relayout):
    rng = _extract_range_from_relayout(relayout)
    if rng is None: raise dash.exceptions.PreventUpdate
    y = _rescale_y_to_window(vix_df, rng)
    if y is None: raise dash.exceptions.PreventUpdate
    p = Patch()
    p["layout"]["yaxis"]["range"]     = y
    p["layout"]["yaxis"]["autorange"] = False
    return p


@app.callback(
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Input("spx-graph", "relayoutData"),
    Input("vix-graph", "relayoutData"),
    prevent_initial_call=True,
)
def sync_dates_from_charts(spx_relayout, vix_relayout):
    relayout = spx_relayout if ctx.triggered_id == "spx-graph" else vix_relayout
    rng = _extract_range_from_relayout(relayout)
    if rng is None:
        raise dash.exceptions.PreventUpdate
    try:
        ts0 = pd.Timestamp(rng[0])
        ts1 = pd.Timestamp(rng[1])
        # Guard against absurd timestamps Plotly emits on first render / autorange
        if ts0.year < 1900 or ts0.year > 2200 or ts1.year < 1900 or ts1.year > 2200:
            raise dash.exceptions.PreventUpdate
        return (snap_to_trading_day(ts0).date(),
                snap_to_trading_day(ts1).date())
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output("drill-dropdown", "value"),
    Input("heatmap-graph", "clickData"),
    prevent_initial_call=True,
)
def heatmap_click_to_drill(click_data):
    if not click_data: raise dash.exceptions.PreventUpdate
    points = click_data.get("points", [])
    if not points: raise dash.exceptions.PreventUpdate
    point  = points[0]
    label  = point.get("label", "")
    parent = point.get("parent", "")
    if parent in ("", "/"):
        if label in GROUP_LIST:
            return label
    else:
        parts = point.get("id", "").split("/")
        if parts and parts[0] in GROUP_LIST:
            return parts[0]
    raise dash.exceptions.PreventUpdate


# Show/hide the sector filter panel
@app.callback(
    Output("all-subind-filter-wrapper", "style"),
    Input("drill-dropdown", "value"),
)
def toggle_subind_filter(drill_value):
    if drill_value == "ALL_SUBIND":
        return {"display": "block", "marginBottom": "8px"}
    return {"display": "none", "marginBottom": "8px"}


@app.callback(
    Output("hidden-subindustries", "data"),
    Input("performance-graph", "clickData"),
    Input("reset-hidden-subind-btn", "n_clicks"),
    State("hidden-subindustries", "data"),
    State("drill-dropdown", "value"),
    prevent_initial_call=True,
)
def manage_hidden_subindustries(click_data, reset_clicks, currently_hidden, drill_value):
    """
    Single-click a sub-industry trace → add it to the hidden list.
    Reset button → clear the hidden list.
    Only active in ALL_SUBIND view; no-ops otherwise.
    """
    triggered = ctx.triggered_id

    if triggered == "reset-hidden-subind-btn":
        return []

    if triggered == "performance-graph" and drill_value == "ALL_SUBIND" and click_data:
        points = click_data.get("points", [])
        if points:
            pt = points[0]
            # customdata[0] is the sub-industry name we embedded on each point
            subind_name = (
                (pt.get("customdata") or [None])[0]
                or pt.get("fullData", {}).get("name")
                or pt.get("name")
            )
            if subind_name and subind_name != "SPX":
                hidden = list(currently_hidden)
                if subind_name not in hidden:
                    hidden.append(subind_name)
                return hidden

    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("hidden-subind-count", "children"),
    Input("hidden-subindustries", "data"),
    State("drill-dropdown", "value"),
)
def update_hidden_count(hidden, drill_value):
    if not hidden:
        return ""
    n = len(hidden)
    return f"{n} sub-industr{'y' if n == 1 else 'ies'} hidden — click a line to hide more"


@app.callback(
    Output("heatmap-graph", "figure"),
    Output("performance-graph", "figure"),
    Input("apply-button", "n_clicks"),
    Input("metric-dropdown", "value"),
    Input("drill-dropdown", "value"),
    Input("sector-visibility", "value"),
    Input("all-subind-sector-filter", "value"),
    Input("hidden-subindustries", "data"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
)
def update_outputs(n_clicks, metric, drill_value, visible_sectors,
                   subind_sectors, hidden_subindustries, start_date, end_date):
    # ── Heatmap ──
    metrics_df = compute_metrics(start_date, end_date, metric,
                                 visible_sectors=visible_sectors or GROUP_LIST)
    s_snap = snap_to_trading_day(pd.Timestamp(start_date)).date()
    e_snap = snap_to_trading_day(pd.Timestamp(end_date)).date()
    label  = next(o["label"] for o in METRIC_OPTIONS if o["value"] == metric)
    label += f"  |  {s_snap} → {e_snap}"
    heatmap_fig = make_treemap(metrics_df, label)

    # ── Performance ──
    if drill_value == "ALL_SUBIND":
        sectors_to_show = subind_sectors or GROUP_LIST
        perf_df  = compute_group_performance(start_date, end_date,
                                             all_subindustries=True,
                                             all_subind_sectors=sectors_to_show)
        perf_fig = make_all_subind_figure(perf_df, sectors_to_show,
                                          hidden_subindustries=hidden_subindustries or [])
    else:
        drill_sector = None if drill_value == "ALL" else drill_value
        perf_df      = compute_group_performance(start_date, end_date, drill_sector)
        perf_fig     = make_performance_figure(perf_df, drill_sector)

    return heatmap_fig, perf_fig


@app.callback(
    Output("wildcard-date-wrapper", "style"),
    Input("top-worst-window", "value"),
)
def toggle_wildcard_picker(window_value):
    if window_value == "wildcard":
        return {"display": "block", "marginBottom": "10px"}
    return {"display": "none", "marginBottom": "10px"}


@app.callback(
    Output("top-worst-table-top", "data"),
    Output("top-worst-table-top", "columns"),
    Output("top-worst-table-top", "style_data_conditional"),
    Output("top-worst-table-worst", "data"),
    Output("top-worst-table-worst", "columns"),
    Output("top-worst-table-worst", "style_data_conditional"),
    Input("top-worst-window", "value"),
    Input("wildcard-date-picker", "date"),
    Input("apply-button", "n_clicks"),
    State("date-range", "end_date"),
)
def update_top_worst(window_value, wildcard_date, n_clicks, end_date):
    wc_date = wildcard_date if window_value == "wildcard" else None

    top_df, worst_df, start_date, latest_date = compute_top_worst_performance(
        end_date, window_value, wildcard_date=wc_date
    )

    top_rows, worst_rows, columns, active_windows = build_top_worst_table_records(
        top_df, worst_df, latest_date, wildcard_date=wc_date
    )
    table_style = make_window_table_style(active_windows, window_value, top_rows, worst_rows)

    return top_rows, columns, table_style, worst_rows, columns, table_style




if __name__ == "__main__":
    app.run(debug=True)

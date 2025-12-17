# streamlit_app.py — Player Minutes Prediction (t -> t+1), EFL League One only (strict)
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ---------------------------- Paths ----------------------------
BASE_DIR   = Path(".").resolve()
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR  = OUTPUT_DIR / "models"

PS_ENH   = OUTPUT_DIR / "players_season_enhanced.parquet"
SUP      = OUTPUT_DIR / "supervised_dataset.parquet"
MODEL_FP = MODEL_DIR / "best_minutes_model.joblib"

# League One scope files (any may exist) — place them in outputs/
LO_PAIRS = OUTPUT_DIR / "league_one_team_seasons.csv"  # preferred: team_id, season_id
LO_IDS   = OUTPUT_DIR / "league_one_team_ids.csv"      # fallback: team_id
LO_TEAMS = OUTPUT_DIR / "league_one_teams.csv"         # your list: team_id and/or team_name

st.set_page_config(page_title="Player Minutes Prediction — EFL League One", layout="wide")

# ---------------------------- Loaders ----------------------------
@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists(): return None
    try: return pd.read_parquet(path)
    except Exception: return None

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists(): return None
    for enc in [None, "utf-8", "latin1"]:
        try: return pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
        except Exception: continue
    return None

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if not path.exists(): return None
    try: return load(path)
    except Exception: return None

ps = load_parquet(PS_ENH)      # player-season data (t)
sup = load_parquet(SUP)        # supervised features with target (t+1)
model = load_model(MODEL_FP)   # trained regressor

# ---------------------------- Canonical team-name helper ----------------------------
def canon_team_name(s: str) -> str:
    x = str(s).lower().strip()
    x = re.sub(r"\b(afc|fc|u21s?|u23s?|reserves?)\b", " ", x)
    x = re.sub(r"[^a-z0-9 ]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

# ---------------------------- Load League One scope ----------------------------
@st.cache_data(show_spinner=False)
def load_league_one_scope():
    """
    Returns (ids, pairs, names_canon):
      - ids:   set[int]           (team_id)
      - pairs: set[(int,int)]     (team_id, season_id)
      - names: set[str]           (canonical team_name)
    """
    ids, pairs, names = set(), set(), set()

    if LO_PAIRS.exists():
        p = load_csv(LO_PAIRS)
        if p is not None and {"team_id","season_id"} <= set(p.columns):
            p = p.dropna(subset=["team_id","season_id"]).copy()
            p["team_id"]   = pd.to_numeric(p["team_id"], errors="coerce").astype("Int64")
            p["season_id"] = pd.to_numeric(p["season_id"], errors="coerce").astype("Int64")
            pairs = set(map(tuple, p[["team_id","season_id"]].drop_duplicates().to_numpy()))

    if LO_IDS.exists():
        d = load_csv(LO_IDS)
        if d is not None and "team_id" in d.columns:
            ids |= set(d["team_id"].dropna().astype("Int64").tolist())

    if LO_TEAMS.exists():
        t = load_csv(LO_TEAMS)
        if t is not None:
            if "team_id" in t.columns:
                ids |= set(t["team_id"].dropna().astype("Int64").tolist())
            if "team_name" in t.columns:
                names |= set(t["team_name"].dropna().astype(str).map(canon_team_name).tolist())

    # normalise empties to None
    ids   = ids   if ids   else None
    pairs = pairs if pairs else None
    names = names if names else None
    return ids, pairs, names

league_one_ids, league_one_pairs, league_one_names = load_league_one_scope()

# Hard gate: if we have none of ids/pairs/names, refuse to show any data
st.sidebar.header("Dataset scope")
if not any([league_one_ids, league_one_pairs, league_one_names]):
    st.sidebar.error(
        "League One scope not loaded.\n"
        "Add one of these to the **outputs/** folder:\n"
        " • league_one_team_seasons.csv  (team_id, season_id)\n"
        " • league_one_team_ids.csv      (team_id)\n"
        " • league_one_teams.csv         (team_id and/or team_name)"
    )
    st.stop()

pairs_ct = len(league_one_pairs) if league_one_pairs else 0
ids_ct   = len(league_one_ids)   if league_one_ids   else 0
names_ct = len(league_one_names) if league_one_names else 0
st.sidebar.success(f"EFL League One")

# ---------------------------- Enforce scope (double gate) ----------------------------
def enforce_league_one(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty or "team_id" not in df.columns:
        return df
    out = df.copy()

    # normalise team name for safe matching
    if "team_name" in out.columns:
        out["team_name"] = out["team_name"].astype(str).str.strip()
        out["team_key"]  = out["team_name"].map(canon_team_name)

    # 1) Prefer team-season pairs
    if league_one_pairs and "season_id" in out.columns:
        key = list(zip(out["team_id"].astype("Int64"), out["season_id"].astype("Int64")))
        mask = pd.Series([tpl in league_one_pairs for tpl in key], index=out.index)
        out = out[mask].copy()

    # 2) Else fallback to team_id list
    elif league_one_ids:
        out = out[out["team_id"].astype("Int64").isin(league_one_ids)].copy()

    # 3) Else fallback to team_name allowlist
    elif league_one_names and "team_key" in out.columns:
        out = out[out["team_key"].isin(league_one_names)].copy()

    # INTERSECTION with names if we have them (extra safety)
    if league_one_names and "team_key" in out.columns:
        out = out[out["team_key"].isin(league_one_names)].copy()

    return out

ps  = enforce_league_one(ps)
sup = enforce_league_one(sup)

# If anything unexpected remains, show diagnostics and drop them
unexpected = []
if ps is not None and not ps.empty and league_one_names and "team_key" in ps.columns:
    bad = ps.loc[~ps["team_key"].isin(league_one_names), "team_name"].dropna().unique().tolist()
    if bad:
        unexpected = sorted(bad)[:10]
        # enforce again: drop unexpected names
        ps = ps[ps["team_key"].isin(league_one_names)].copy()

# st.sidebar.caption(
#     "Teams loaded after filter: "
#     f"{ps['team_name'].nunique() if ps is not None and 'team_name' in ps.columns else 0}"
# )
if unexpected:
    st.sidebar.warning("Removed unexpected teams (sample): " + ", ".join(unexpected))

# ---------------------------- Helpers ----------------------------
def season_maps(df: pd.DataFrame) -> tuple[dict[int,str], dict[str,int], list[int]]:
    if df is None or "season_id" not in df.columns:
        return {}, {}, []
    if "season_name" in df.columns:
        pairs = df[["season_id","season_name"]].dropna().drop_duplicates()
        id_to_name = dict(zip(pairs["season_id"].astype(int), pairs["season_name"].astype(str)))
    else:
        uniq = sorted(df["season_id"].dropna().astype(int).unique())
        id_to_name = {sid: str(sid) for sid in uniq}
    name_to_id = {v: k for k, v in id_to_name.items()}
    ordered = sorted(id_to_name.keys())
    return id_to_name, name_to_id, ordered

def next_season_id(current_id: int, ordered_ids: list[int]) -> int | None:
    if current_id is None or current_id not in ordered_ids: return None
    i = ordered_ids.index(current_id)
    return ordered_ids[i+1] if i+1 < len(ordered_ids) else None

def pick_feature_cols(sup_df: pd.DataFrame) -> list[str]:
    return [c for c in sup_df.columns if c not in ["player_id","season_id","target_minutes_next_season"]]

id_to_name, name_to_id, ordered_ids = season_maps(ps if ps is not None else sup)

# ---------------------------- Sidebar Filters ----------------------------
st.sidebar.header("Filters")

team_choice = "(All)"
pos_choice  = "(All)"
season_label = "(All)"
name_q = ""

if ps is not None and not ps.empty:
    teams = ["(All)"] + sorted(ps["team_name"].dropna().astype(str).str.strip().unique().tolist())
    team_choice = st.sidebar.selectbox("Team", teams)

    poss = ["(All)"] + sorted([p for p in ps["primary_position"].dropna().unique().tolist() if p])
    pos_choice = st.sidebar.selectbox("Position", poss)

    season_pairs = (ps[["season_id","season_name"]].dropna().drop_duplicates().sort_values("season_id")
                    if {"season_id","season_name"} <= set(ps.columns) else pd.DataFrame())
    season_labels = ["(All)"] + (season_pairs["season_name"].tolist() if not season_pairs.empty else [])
    season_label  = st.sidebar.selectbox("Season", season_labels)

    name_q = st.sidebar.text_input("Search player name")

def apply_filters(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty: return df
    out = df.copy()
    if team_choice != "(All)" and "team_name" in out.columns:
        out = out[out["team_name"].astype(str).str.strip() == team_choice]
    if pos_choice != "(All)" and "primary_position" in out.columns:
        out = out[out["primary_position"] == pos_choice]
    if season_label != "(All)" and not season_pairs.empty:
        sid = int(season_pairs.loc[season_pairs["season_name"] == season_label, "season_id"].iloc[0])
        out = out[out["season_id"] == sid]
    if name_q and "player_name" in out.columns:
        out = out[out["player_name"].str.contains(name_q, case=False, na=False)]
    return out

# ---------------------------- UI ----------------------------
st.title("⚽ Player Minutes Prediction — Season (EFL League One)")

st.subheader("Player–Season records")
if ps is None or ps.empty:
    st.error("No rows after League One enforcement. Check that your scope file is in **outputs/**.")
else:
    view_cols = [c for c in [
        "player_id","player_name","team_name","primary_position",
        "season_name","player_match_minutes","player_match_goals","player_match_assists"
    ] if c in ps.columns]
    st.dataframe(apply_filters(ps)[view_cols].head(400), use_container_width=True)

st.markdown("---")
st.header("Predict next-season minutes for a selected player–season")

if model is None or sup is None or sup.empty:
    st.info("Model or supervised dataset not found. Ensure files exist in outputs/.")
else:
    if ps is None or ps.empty:
        st.warning("No League One player-season rows available to select.")
    else:
        id_cols = ["player_id","season_id"]
        feat_cols = pick_feature_cols(sup)

        candidates = ps.merge(sup[id_cols].drop_duplicates(), on=id_cols, how="inner")
        candidates = apply_filters(candidates)

        if candidates.empty:
            st.info("No matching rows with current filters. Adjust filters.")
        else:
            def row_label(r):
                bits = [str(r.get("player_name",""))]
                if "team_name" in r: bits.append(str(r.get("team_name","")))
                if "season_name" in r: bits.append(str(r.get("season_name","")))
                return " — ".join([b for b in bits if b])

            opts = candidates[["player_id","season_id","player_name","team_name","season_name"]].drop_duplicates().reset_index(drop=True)
            index_choice = st.selectbox("Choose player-season (season t)", opts.index,
                                        format_func=lambda i: row_label(opts.loc[i]))
            sel = opts.loc[index_choice, ["player_id","season_id"]]

            feat_row = sup[(sup["player_id"] == sel["player_id"]) & (sup["season_id"] == sel["season_id"])]
            if feat_row.empty:
                st.warning("Could not find feature row in supervised dataset.")
            else:
                # Predict t+1
                y_pred = float(model.predict(feat_row[feat_cols].values)[0])
                st.metric("Predicted minutes (t+1)", f"{y_pred:,.0f}")

                # Lookup actual t+1 using chronological season order
                id_to_name, name_to_id, ordered_ids = season_maps(ps)
                sid_next = next_season_id(int(sel["season_id"]), ordered_ids)
                y_true = None
                if sid_next is not None:
                    nxt = ps[(ps["player_id"] == sel["player_id"]) & (ps["season_id"] == sid_next)]
                    if not nxt.empty and "player_match_minutes" in nxt.columns:
                        y_true = float(nxt["player_match_minutes"].iloc[0])
                if y_true is not None:
                    st.metric("Actual minutes (t+1)", f"{y_true:,.0f}", delta=f"{(y_pred - y_true):+.0f} (pred-actual)")
                else:
                    st.caption("No actual t+1 minutes available (likely latest season).")

                # What-if (minimal)
                with st.expander("🔧 What-if: tweak a few inputs for this player-season", expanded=False):
                    x_base = feat_row[feat_cols].iloc[0].copy()
                    tweakable = [f for f in [
                        "player_match_minutes",
                        "goals_per90","assists_per90","np_shots_per90","key_passes_per90"
                    ] if f in feat_row.columns]

                    def slider_range(col):
                        s = sup[col].dropna()
                        if s.empty: return (0.0, 1.0, float(x_base.get(col, 0.0)))
                        lo, hi = np.quantile(s, [0.01, 0.99])
                        val = float(x_base.get(col, np.nan))
                        if np.isnan(val): val = float(np.clip(s.median(), lo, hi))
                        return float(lo), float(hi), float(val)

                    cols2 = st.columns(2)
                    new_vals = {}
                    for i, col in enumerate(tweakable):
                        lo, hi, val = slider_range(col)
                        step = 1.0 if "minutes" in col else 0.01
                        new_vals[col] = cols2[i % 2].slider(col, min_value=lo, max_value=hi, value=val, step=step)

                    x_mod = x_base.copy()
                    for k, v in new_vals.items():
                        x_mod[k] = v
                    X0 = np.asarray([x_base.reindex(feat_cols).to_numpy()])
                    X1 = np.asarray([x_mod.reindex(feat_cols).to_numpy()])
                    y0 = float(model.predict(X0)[0])
                    y1 = float(model.predict(X1)[0])

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Baseline prediction", f"{y0:,.0f}")
                    c2.metric("What-if prediction", f"{y1:,.0f}", delta=f"{(y1 - y0):+.0f}")

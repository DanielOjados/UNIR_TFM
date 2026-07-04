"""
Stage 02 – Build Master Dataset
================================
Joins the reconstructed participantes table with all dimension tables to
produce three versioned master datasets:

  v1_baseline    – participantes + carreras + jornadas core fields
  v2_performance – v1 + caballos/jinetes/preparadores info aggregates
  v3_full        – v2 + genealogy + pista/surface from carreras

Then performs a temporal train/val/test split and saves all outputs.

Run:
    python pipelines/02_build_master_dataset.py
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import (
    INTERIM_FILES,
    LOG_FORMAT,
    LOG_LEVEL,
    PROCESSED_FILES,
    RAW_FILES,
    TRAIN_END_DATE,
    VAL_END_DATE,
)
from bucephalus.io_utils import load_parquet, save_parquet, save_csv
from bucephalus.temporal import temporal_split

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger("02_build_master_dataset")


# ── Column selection per version ──────────────────────────────────────────────

V1_CARRERA_COLS = [
    "carrera_id", "jornada_id", "numero", "nombre", "condiciones",
    "participantes", "distancia", "pista",
]
V1_JORNADA_COLS = ["jornada_id", "hipodromo", "fecha", "euros"]


# ── Builders ──────────────────────────────────────────────────────────────────

def load_participantes_merged() -> "pd.DataFrame":
    """
    Load participantes_merged as the authoritative source for:
      - pos_pm   : finish position (100% coverage — primary source for targets)
      - odds_win : pre-race win dividend (~65% coverage)
      - dist_al_anterior : distance to previous horse (post-race, used as context)

    participantes_merged is more reliable than caballos_historial for positions:
    caballos_historial has 23% null pos (mostly pre-2005 scraping gap).
    """
    path = RAW_FILES["participantes_merged"]
    if not path.exists():
        logger.warning("participantes_merged not found at %s — returning empty", path)
        return pd.DataFrame(columns=["carrera_id", "caballo_id",
                                      "pos_pm", "odds_win", "distancia_al_anterior"])
    df = pd.read_csv(path, sep=";", encoding="latin1", low_memory=False)
    df = df.rename(columns={
        "posición"          : "pos_pm",       # PRIMARY position source
        "dividendo_ganador" : "odds_win",
        "CajonDeSalida"     : "casilla",
    })
    df["pos_pm"]  = pd.to_numeric(df["pos_pm"],  errors="coerce")
    df["odds_win"] = pd.to_numeric(df["odds_win"], errors="coerce")
    # Odds must be > 1; values ≤ 1 in the raw CSV represent missing/unpublished odds.
    df["odds_win"] = df["odds_win"].where(df["odds_win"] > 1, np.nan)
    df["distancia_al_anterior"] = pd.to_numeric(df["distancia_al_anterior"], errors="coerce")
    df["peso"] = pd.to_numeric(df["peso"], errors="coerce")
    df["jinete_id"]     = pd.to_numeric(df["jinete_id"],     errors="coerce")
    df["preparador_id"] = pd.to_numeric(df["preparador_id"], errors="coerce")
    slim = df[[
        "carrera_id", "caballo_id",
        "pos_pm", "peso", "casilla",
        "jinete_id", "preparador_id",
        "odds_win", "distancia_al_anterior",
    ]].drop_duplicates(subset=["carrera_id", "caballo_id"])
    n_pos  = slim["pos_pm"].notna().sum()
    n_odds = slim["odds_win"].notna().sum()
    n_peso = slim["peso"].notna().sum()
    logger.info(
        "participantes_merged: %d rows | pos_pm %d (%.1f%%) | peso %d (%.1f%%) | odds_win %d (%.1f%%)",
        len(slim), n_pos, 100 * n_pos / max(len(slim), 1),
        n_peso, 100 * n_peso / max(len(slim), 1),
        n_odds, 100 * n_odds / max(len(slim), 1),
    )
    return slim


def load_participantes_odds() -> "pd.DataFrame":
    """Backward-compat shim: returns slim odds-only table (used by patch scripts)."""
    pm = load_participantes_merged()
    return pm[["carrera_id", "caballo_id", "odds_win", "distancia_al_anterior"]]


def build_v1(
    participantes: pd.DataFrame,
    carreras: pd.DataFrame,
    jornadas: pd.DataFrame,
) -> pd.DataFrame:
    """
    v1_baseline: one row per horse per race with core race/meeting context.
    """
    # Bring in carrera-level columns (nome, condiciones, distancia, pista)
    car_slim = carreras[[c for c in V1_CARRERA_COLS if c in carreras.columns]].drop_duplicates("carrera_id")
    jor_slim = jornadas[[c for c in V1_JORNADA_COLS if c in jornadas.columns]].drop_duplicates("jornada_id")

    v1 = participantes.copy()
    # Drop rows unlinked to a carrera — they break the (caballo_id, carrera_id) composite PK
    before = len(v1)
    if "carrera_id" in v1.columns:
        v1 = v1[v1["carrera_id"].notna()].copy()
        logger.info(
            "Dropped %d rows with null carrera_id (%.1f%%) — kept in interim/participantes only",
            before - len(v1), 100 * (before - len(v1)) / before,
        )
    # If carrera_id was resolved, merge carrera details; otherwise keep historial fields
    if "carrera_id" in v1.columns:
        v1 = v1.merge(car_slim, on="carrera_id", how="left", suffixes=("", "_car"))
    v1 = v1.merge(jor_slim, on="jornada_id", how="left", suffixes=("", "_jor"))

    # Normalise fecha: prefer jornada fecha when available, fall back to historial fecha
    if "fecha_jor" in v1.columns:
        v1["fecha"] = v1["fecha"].fillna(v1["fecha_jor"])
        v1 = v1.drop(columns=["fecha_jor"], errors="ignore")
    if "hipodromo_jor" in v1.columns:
        v1["hipodromo"] = v1["hipodromo"].fillna(v1["hipodromo_jor"])
        v1 = v1.drop(columns=["hipodromo_jor"], errors="ignore")

    # ── Merge + extend from participantes_merged ─────────────────────────────
    # participantes_merged is the AUTHORITATIVE race-participant list.
    # 1. Enrich historial rows with pm.pos / pm.peso (left join).
    # 2. Add SKELETON ROWS for pm-only entries (winner not in historial) so that
    #    every race has its winner represented → race_has_winner=1 everywhere.
    from bucephalus.features import add_targets
    pm = load_participantes_merged()
    if len(pm) > 0:
        # -- Step 1: enrich existing historial rows --
        pm_enrich = pm.rename(columns={"peso": "peso_pm", "casilla": "casilla_pm",
                                        "jinete_id": "jinete_id_pm",
                                        "preparador_id": "preparador_id_pm"})
        v1 = v1.merge(pm_enrich, on=["carrera_id", "caballo_id"], how="left")
        n_pm_pos = v1["pos_pm"].notna().sum()
        v1["pos"]  = v1["pos_pm"].combine_first(v1["pos"])
        v1["peso"] = v1["peso_pm"].combine_first(v1.get("peso", pd.Series(dtype=float)))
        # Fill missing jinete/preparador from pm (historial may have NaN for some)
        if "jinete_id" in v1.columns:
            v1["jinete_id"] = v1["jinete_id"].combine_first(v1["jinete_id_pm"])
        if "preparador_id" in v1.columns:
            v1["preparador_id"] = v1["preparador_id"].combine_first(v1["preparador_id_pm"])
        v1 = v1.drop(columns=["pos_pm", "peso_pm", "casilla_pm",
                               "jinete_id_pm", "preparador_id_pm"], errors="ignore")
        logger.info(
            "pm enrichment: %d rows with pos_pm | %d with odds_win",
            n_pm_pos, v1["odds_win"].notna().sum(),
        )

        # -- Step 2: skeleton rows for pm-only horse-race combinations --
        v1_keys = set(zip(v1["carrera_id"].tolist(), v1["caballo_id"].tolist()))
        pm_extra = pm[
            ~pd.Series(list(zip(pm["carrera_id"].tolist(),
                                 pm["caballo_id"].tolist())))
            .isin(v1_keys).values
        ].copy()
        if len(pm_extra) > 0:
            skel = pm_extra.rename(columns={"pos_pm": "pos"})
            # Add carrera and jornada context
            skel = skel.merge(car_slim, on="carrera_id", how="left")
            skel = skel.merge(jor_slim, on="jornada_id", how="left", suffixes=("", "_jor"))
            if "fecha_jor" in skel.columns:
                skel["fecha"] = skel["fecha"].fillna(skel["fecha_jor"])
                skel = skel.drop(columns=["fecha_jor"], errors="ignore")
            if "hipodromo_jor" in skel.columns:
                skel["hipodromo"] = skel.get("hipodromo", pd.Series(dtype=str)).fillna(skel["hipodromo_jor"])
                skel = skel.drop(columns=["hipodromo_jor"], errors="ignore")
            v1 = pd.concat([v1, skel], ignore_index=True, sort=False)
            logger.info(
                "skeleton rows added from pm-only entries: %d (races with no historial coverage)",
                len(pm_extra),
            )

    # Add target labels (uses updated pos — participantes_merged is primary source)
    v1 = add_targets(v1, pos_col="pos")

    logger.info("v1_baseline: %d rows × %d cols", len(v1), v1.shape[1])
    return v1


def build_v2(
    v1: pd.DataFrame,
    jinetes_info: pd.DataFrame,
    preparadores_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    v2_performance: v1 + jinete_id / preparador_id identity columns.

    NOTE: static career totals (jinetes_info.victorias / preparadores_info.victorias)
    are NOT merged here because they are snapshot values from the data-dump date
    (2025-03-29) and would assign future win counts to historical races → temporal
    leakage.  Correct temporal equivalents are computed in Stage 04 via
    expanding_stats(shift=1): jockey_win_rate_cum, jockey_n_prev_races,
    trainer_win_rate_cum, trainer_n_prev_races, etc.
    """
    v2 = v1.copy()
    logger.info("v2_performance: %d rows × %d cols", len(v2), v2.shape[1])
    return v2


def build_v3(
    v2: pd.DataFrame,
    genealogy: pd.DataFrame,
) -> pd.DataFrame:
    """
    v3_full: v2 + genealogy dimension fields.
    Genealogy performance features (sire/dam offspring stats) are added in Stage 04.
    """
    v3 = v2.copy()
    if "caballo_id" in v3.columns:
        gen_slim = genealogy[["caballo_id", "padre_id", "madre_id",
                               "criador_id", "padre___madre"]].drop_duplicates("caballo_id")
        v3 = v3.merge(gen_slim, on="caballo_id", how="left")

    logger.info("v3_full: %d rows × %d cols", len(v3), v3.shape[1])
    return v3


# ── Summary stats helper ───────────────────────────────────────────────────────

def _print_target_stats(df: pd.DataFrame, label: str) -> None:
    for t in ["target_top1", "target_top3"]:
        if t in df.columns:
            rate = df[t].mean()
            logger.info(
                "[%s] %s positive rate: %.3f (%.1f%%)", label, t, rate, 100 * rate
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("STAGE 02 – BUILD MASTER DATASET")
    logger.info("=" * 60)

    # ── Load cleaned tables ───────────────────────────────────────────────────
    logger.info("Loading interim parquets …")
    participantes     = load_parquet(INTERIM_FILES["participantes"])
    carreras          = load_parquet(INTERIM_FILES["carreras"])
    jornadas          = load_parquet(INTERIM_FILES["jornadas"])
    jinetes_info      = load_parquet(INTERIM_FILES["jinetes_info"])
    preparadores_info = load_parquet(INTERIM_FILES["preparadores_info"])
    genealogy         = load_parquet(INTERIM_FILES["genealogy"])

    logger.info(
        "Participantes loaded: %d rows | date range %s → %s",
        len(participantes),
        participantes["fecha"].min(),
        participantes["fecha"].max(),
    )

    # ── Build master versions ─────────────────────────────────────────────────
    v1 = build_v1(participantes, carreras, jornadas)
    _print_target_stats(v1, "v1")
    save_parquet(v1, PROCESSED_FILES["v1_baseline"])
    save_csv(v1, PROCESSED_FILES["v1_baseline"].with_suffix(".csv"))

    v2 = build_v2(v1, jinetes_info, preparadores_info)
    _print_target_stats(v2, "v2")
    save_parquet(v2, PROCESSED_FILES["v2_performance"])

    v3 = build_v3(v2, genealogy)
    _print_target_stats(v3, "v3")
    save_parquet(v3, PROCESSED_FILES["v3_full"])

    # ── Temporal split (on v3 – the richest version) ─────────────────────────
    logger.info(
        "Performing temporal split: train < %s | val %s–%s | test ≥ %s",
        TRAIN_END_DATE, TRAIN_END_DATE, VAL_END_DATE, VAL_END_DATE,
    )
    train, val, test = temporal_split(
        v3, date_col="fecha",
        train_end=TRAIN_END_DATE,
        val_end=VAL_END_DATE,
    )

    # Sanity: no target leakage across splits
    if train["fecha"].max() >= pd.Timestamp(TRAIN_END_DATE):
        logger.error("LEAKAGE: train set contains rows with fecha >= %s", TRAIN_END_DATE)
    if val["fecha"].max() >= pd.Timestamp(VAL_END_DATE):
        logger.error("LEAKAGE: val set contains rows with fecha >= %s", VAL_END_DATE)

    _print_target_stats(train, "train")
    _print_target_stats(val,   "val")
    _print_target_stats(test,  "test")

    save_parquet(train, PROCESSED_FILES["train"])
    save_parquet(val,   PROCESSED_FILES["val"])
    save_parquet(test,  PROCESSED_FILES["test"])

    # ── Column summary ────────────────────────────────────────────────────────
    logger.info("Column counts: v1=%d | v2=%d | v3=%d", v1.shape[1], v2.shape[1], v3.shape[1])
    logger.info("All master datasets written to: %s", PROCESSED_FILES["v3_full"].parent)
    logger.info("Stage 02 complete.")


if __name__ == "__main__":
    main()

"""
Stage 04 – Feature Engineering
================================
Loads v3_full master dataset and applies all leakage-safe rolling/expanding
feature groups.  Outputs enriched parquet files and a final
train/val/test split ready for modelling.

Feature groups built:
  1.  Horse form (rolling pos, win rate, top3 rate, earnings)
  2.  Jockey form
  3.  Trainer form
  4.  Horse–jockey synergy
  5.  Horse–trainer synergy
  6.  Distance suitability
  7.  Track / hipodrome suitability
  8.  Rest days
  9.  Within-race relative features (rank, percentile, z-score)
  10. Genealogy aggregated features
  11. Field-relative features — vs_field_* (Lever 2, docs/modeling_strategy.md)
  12. Form trajectory slope — horse_pos_trend_* (Lever 6)
  13. Form cycle / fatigue — horse_races_last_*d (Lever 10)
  14. Jockey change signal — jockey_upgrade (Lever 11)
  15. Distance step-up/step-down — distance_change_* (Lever 12)

Run:
    python pipelines/04_feature_engineering.py
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import (
    INTERIM_FILES,
    LOG_FORMAT,
    LOG_LEVEL,
    PROCESSED_DIR,
    PROCESSED_FILES,
    ROLLING_WINDOWS,
    TRAIN_END_DATE,
    VAL_END_DATE,
)
from bucephalus.features import (
    add_distance_change,
    add_distance_suitability,
    add_field_relative_features,
    add_form_cycle,
    add_form_trajectory,
    add_genealogy_features,
    add_horse_form,
    add_horse_jockey_synergy,
    add_horse_trainer_synergy,
    add_jockey_change_signal,
    add_jockey_form,
    add_surface_suitability,
    add_targets,
    add_track_suitability,
    add_trainer_form,
    add_within_race_relative,
)
from bucephalus.io_utils import load_parquet, save_parquet, save_csv
from bucephalus.temporal import temporal_split

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger("04_feature_engineering")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_feature_count(df: pd.DataFrame, stage: str) -> None:
    logger.info("[%s] shape: %d rows × %d cols", stage, len(df), df.shape[1])


# Columns that are POST-RACE outcomes — must NEVER appear in model features.
# pos = actual finish position (the label); pos_dnf = derived DNF flag;
# euros = prize money won in this specific race.
_OUTCOME_COLS = {"pos", "pos_dnf", "euros", "distancias", "tiempo", "premio_eur"}


def _drop_raw_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove raw helper columns and any post-race outcome columns that must not leak into features."""
    to_drop = [
        c for c in df.columns
        if c.endswith("_raw") or c in ("extra", "mant") or c in _OUTCOME_COLS
        or c == "jornada_id_car"   # join artifact: duplicate of jornada_id
    ]
    return df.drop(columns=to_drop, errors="ignore")


# ── Main feature pipeline ─────────────────────────────────────────────────────

def build_features(
    v3: pd.DataFrame,
    caballos_hist: pd.DataFrame,
    genealogy: pd.DataFrame,
    windows: list,
) -> pd.DataFrame:
    """
    Apply all feature groups to v3_full.
    All rolling features are computed on v3 itself (which represents the
    full longitudinal race history per horse/jockey/trainer) in chronological
    order.  The shift(1) inside each rolling function prevents leakage.
    """
    df = v3.copy()
    _log_feature_count(df, "input v3")

    # ── Ensure pos is numeric ─────────────────────────────────────────────────
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")

    # ── Clean peso (handicap asignado — dato PRE-carrera válido) ─────────────
    # El hándicap asignado se publica en el programa oficial días antes de la
    # carrera → es información pre-race, sin leakage.
    # Se limpian outliers obvios de entrada de datos (>100 kg o ==0).
    # Rango normal en carreras españolas: 50–75 kg (p99 = 67 kg).
    if "peso" in df.columns:
        df["peso"] = pd.to_numeric(df["peso"], errors="coerce")
        n_outliers = ((df["peso"] < 40) | (df["peso"] > 100)).sum()
        df["peso"] = df["peso"].where((df["peso"] >= 40) & (df["peso"] <= 100), np.nan)
        logger.info("peso: %d outliers (< 40 kg or > 100 kg) → NaN (%.2f%%)",
                    n_outliers, 100 * n_outliers / max(len(df), 1))

    # ── 1. Horse form ─────────────────────────────────────────────────────────
    logger.info("Computing horse form features …")
    df = add_horse_form(df, windows=windows)
    _log_feature_count(df, "after horse form")

    # ── 2. Jockey form ────────────────────────────────────────────────────────
    logger.info("Computing jockey form features …")
    df = add_jockey_form(df, windows=windows)
    _log_feature_count(df, "after jockey form")

    # ── 3. Trainer form ───────────────────────────────────────────────────────
    logger.info("Computing trainer form features …")
    df = add_trainer_form(df, windows=windows)
    _log_feature_count(df, "after trainer form")

    # ── 3b. Career stats — canonical aliases for the temporal columns above ──
    # jockey_n_prev_races / jockey_cum_wins / jockey_cum_earnings / jockey_win_rate_cum
    # are already computed correctly with shift(1) in steps 2/3.
    # Expose them under the canonical "career_" name for model compatibility.
    # NOTE: jinetes_hist/preparadores_hist are NOT used here — they only contain
    # ~10 entries per entity (last scraped races) giving <15% coverage.
    # The master dataset itself is the complete career history.
    CAREER_MAP = {
        "jockey_career_races"    : "jockey_n_prev_races",
        "jockey_career_wins"     : "jockey_cum_wins",
        "jockey_career_earnings" : "jockey_cum_earnings",
        "jockey_career_win_rate" : "jockey_win_rate_cum",
        "trainer_career_races"   : "trainer_n_prev_races",
        "trainer_career_wins"    : "trainer_cum_wins",
        "trainer_career_earnings": "trainer_cum_earnings",
        "trainer_career_win_rate": "trainer_win_rate_cum",
    }
    for career_col, src_col in CAREER_MAP.items():
        if src_col in df.columns:
            df[career_col] = df[src_col]
    _log_feature_count(df, "after career stat aliases")

    # ── 4. Horse–jockey synergy ───────────────────────────────────────────────
    logger.info("Computing horse–jockey synergy …")
    df = add_horse_jockey_synergy(df, windows=windows)
    _log_feature_count(df, "after hj synergy")

    # ── 5. Horse–trainer synergy ──────────────────────────────────────────────
    logger.info("Computing horse–trainer synergy …")
    df = add_horse_trainer_synergy(df, windows=windows)
    _log_feature_count(df, "after ht synergy")

    # ── 6. Distance suitability ───────────────────────────────────────────────
    logger.info("Computing distance suitability …")
    df = add_distance_suitability(df, windows=windows)
    _log_feature_count(df, "after distance suitability")

    # ── 7. Track / hipodrome suitability ──────────────────────────────────────
    logger.info("Computing track suitability …")
    df = add_track_suitability(df, windows=windows)
    if "pista" in df.columns:
        df = add_surface_suitability(df, windows=windows)
    _log_feature_count(df, "after track suitability")

    # ── 8. Within-race relative features ──────────────────────────────────────
    logger.info("Computing within-race relative features …")
    if "carrera_id" in df.columns:
        df = add_within_race_relative(df, race_col="carrera_id")
    _log_feature_count(df, "after within-race features")

    # ── 9. Genealogy features ─────────────────────────────────────────────────
    logger.info("Computing genealogy features …")
    df = add_genealogy_features(df, genealogy=genealogy, caballos_hist=caballos_hist)
    _log_feature_count(df, "after genealogy features")

    # ── 10. Ensure targets present ────────────────────────────────────────────
    if "target_top1" not in df.columns:
        df = add_targets(df)

    # ── 11. Field-relative features (Lever 2) ────────────────────────────────
    logger.info("Computing field-relative features (Lever 2) …")
    if "carrera_id" in df.columns:
        df = add_field_relative_features(df, race_col="carrera_id")
    _log_feature_count(df, "after field-relative")

    # ── 12. Form trajectory slope (Lever 6 lightweight) ──────────────────────
    logger.info("Computing form trajectory slope (Lever 6) …")
    df = add_form_trajectory(df, windows=[5, 10])
    _log_feature_count(df, "after form trajectory")

    # ── 13. Form cycle / fatigue (Lever 10) ───────────────────────────────────
    logger.info("Computing form cycle / fatigue features (Lever 10) …")
    df = add_form_cycle(df)
    _log_feature_count(df, "after form cycle")

    # ── 14. Jockey change signal (Lever 11) ───────────────────────────────────
    logger.info("Computing jockey change signal (Lever 11) …")
    df = add_jockey_change_signal(df)
    _log_feature_count(df, "after jockey change signal")

    # ── 15. Distance step-up/step-down (Lever 12) ─────────────────────────────
    logger.info("Computing distance change features (Lever 12) …")
    df = add_distance_change(df)
    _log_feature_count(df, "after distance change")

    # ── Drop raw helper columns ───────────────────────────────────────────────
    df = _drop_raw_cols(df)

    return df


# ── Feature importance helper (basic variance filter) ─────────────────────────

def feature_variance_report(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Report variance of all numeric features – useful for zero-variance removal."""
    exclude = {"target_top1", "target_top3", "target_pos", "target_rank_label", "caballo_id",
               "jinete_id", "preparador_id", "carrera_id", "jornada_id"}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    stats = pd.DataFrame({
        "feature"    : num_cols,
        "non_null_pct": [round(100 * df[c].notna().mean(), 2) for c in num_cols],
        "variance"   : [float(df[c].var())                   for c in num_cols],
        "mean"       : [float(df[c].mean())                  for c in num_cols],
        "std"        : [float(df[c].std())                   for c in num_cols],
    }).sort_values("variance", ascending=False)
    stats.to_csv(output_path, index=False)
    logger.info("Feature variance report saved: %s (%d features)", output_path, len(stats))
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("STAGE 04 – FEATURE ENGINEERING")
    logger.info("=" * 60)
    logger.info("Rolling windows: %s", ROLLING_WINDOWS)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading v3_full master dataset …")
    v3                = load_parquet(PROCESSED_FILES["v3_full"])
    caballos_hist     = load_parquet(INTERIM_FILES["caballos_hist"])
    genealogy         = load_parquet(INTERIM_FILES["genealogy"])

    logger.info("v3 shape: %d × %d | date range: %s → %s",
                len(v3), v3.shape[1],
                v3["fecha"].min(), v3["fecha"].max())

    # ── Build features ────────────────────────────────────────────────────────
    df_fe = build_features(v3, caballos_hist, genealogy, windows=ROLLING_WINDOWS)

    logger.info("Feature engineering complete: %d rows × %d cols", len(df_fe), df_fe.shape[1])

    # ── Temporal split on enriched dataset ───────────────────────────────────
    train, val, test = temporal_split(
        df_fe, date_col="fecha",
        train_end=TRAIN_END_DATE,
        val_end=VAL_END_DATE,
    )

    # ── Save enriched versions ────────────────────────────────────────────────
    fe_dir = PROCESSED_DIR
    save_parquet(df_fe,  fe_dir / "master_v3_full_features.parquet")
    save_parquet(train,  fe_dir / "feat_train.parquet")
    save_parquet(val,    fe_dir / "feat_val.parquet")
    save_parquet(test,   fe_dir / "feat_test.parquet")
    save_csv(df_fe.head(5000), fe_dir / "master_v3_full_features_sample.csv")

    logger.info(
        "Splits: train=%d | val=%d | test=%d",
        len(train), len(val), len(test),
    )

    # ── Feature report ────────────────────────────────────────────────────────
    from config.settings import REPORTS_DIR
    feature_variance_report(df_fe, REPORTS_DIR / "feature_variance_report.csv")

    # ── Feature inventory ─────────────────────────────────────────────────────
    feature_cols = [
        c for c in df_fe.columns
        if c not in {
            "caballo_id", "jinete_id", "preparador_id", "carrera_id",
            "jornada_id", "fecha", "hipodromo", "carrera_nombre",
            "jinete_nombre", "preparador_nombre", "numero", "nombre",
            "target_top1", "target_top3", "target_pos", "target_rank_label",
            "jornada_id_car",
        }
    ]
    inventory = pd.DataFrame({
        "feature"   : feature_cols,
        "dtype"     : [str(df_fe[c].dtype) for c in feature_cols],
        "null_pct"  : [round(100 * df_fe[c].isna().mean(), 2) for c in feature_cols],
        "group"     : [
            "trajectory"     if "pos_trend" in c else
            "form_cycle"     if "races_last_" in c else
            "jockey_signal"  if c == "jockey_upgrade" else
            "distance_delta" if c.startswith("distance_change") else
            "field_relative" if c.startswith("vs_field_") or c in
                                {"field_strength_avg_wr","field_n_runners",
                                 "casilla_imputed","casilla_rel"} else
            "horse_form"     if c.startswith("horse_") else
            "jockey_form"    if c.startswith("jockey_") else
            "trainer_form"   if c.startswith("trainer_") else
            "hj_synergy"     if c.startswith("hj_") else
            "ht_synergy"     if c.startswith("ht_") else
            "within_race"    if c.startswith("wr_") else
            "genealogy"      if c in {"padre_id","madre_id","criador_id","padre___madre",
                                      "sire_offspring_avg_pos","dam_offspring_avg_pos",
                                      "sire_offspring_races","dam_offspring_races",
                                      "sire_offspring_top3_rate","dam_offspring_top3_rate"} else
            "race_context"   if c in {"distancia_m","pista_recta","peso","casilla",
                                      "condiciones","pista","distancia"} else
            "other"
            for c in feature_cols
        ],
    })
    inventory.to_csv(REPORTS_DIR / "feature_inventory.csv", index=False)
    logger.info(
        "Feature inventory saved (%d features). Group summary:\n%s",
        len(inventory),
        inventory.groupby("group")["feature"].count().to_string(),
    )

    logger.info("Stage 04 complete.")


if __name__ == "__main__":
    main()

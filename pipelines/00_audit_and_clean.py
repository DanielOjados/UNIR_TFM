"""
Stage 00 – Audit & Clean
========================
Loads every raw source file, applies standardisation/cleaning, runs a
data-quality audit, and writes cleaned parquet files to data/interim/.

Run:
    python pipelines/00_audit_and_clean.py
"""

import logging
import sys
from pathlib import Path

# ── Path bootstrap (allows running from project root or pipelines/) ───────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    INTERIM_FILES,
    LOG_FORMAT,
    LOG_LEVEL,
    RAW_FILES,
    REPORTS_DIR,
)
from bucephalus.cleaning import (
    clean_caballos_hist,
    clean_carreras,
    clean_genealogy,
    clean_jinetes_hist,
    clean_jinetes_info,
    clean_jornadas,
    clean_preparadores_hist,
    clean_preparadores_info,
)
from bucephalus.io_utils import load_csv, load_excel, save_parquet
from bucephalus.reporting import (
    full_quality_report,
    generate_data_dictionary,
    null_report,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger("00_audit_and_clean")


# ── Individual loaders/cleaners ───────────────────────────────────────────────

def load_and_clean_jornadas() -> "pd.DataFrame":
    import pandas as pd
    raw = load_csv(RAW_FILES["jornadas"])
    df  = clean_jornadas(raw)
    save_parquet(df, INTERIM_FILES["jornadas"])
    logger.info("jornadas: date range %s → %s", df["fecha"].min(), df["fecha"].max())
    return df


def load_and_clean_carreras() -> "pd.DataFrame":
    raw = load_csv(RAW_FILES["carreras"])
    df  = clean_carreras(raw)
    save_parquet(df, INTERIM_FILES["carreras"])
    return df


def load_and_clean_caballos_hist() -> "pd.DataFrame":
    import pandas as pd
    raw = load_csv(RAW_FILES["caballos_hist"])
    df  = clean_caballos_hist(raw)
    save_parquet(df, INTERIM_FILES["caballos_hist"])
    logger.info(
        "caballos_hist: %d rows | %d unique horses | date range %s → %s",
        len(df), df["caballo_id"].nunique(),
        df["fecha"].min(), df["fecha"].max(),
    )
    return df


def load_and_clean_jinetes_info() -> "pd.DataFrame":
    raw = load_csv(RAW_FILES["jinetes_info"])
    df  = clean_jinetes_info(raw)
    save_parquet(df, INTERIM_FILES["jinetes_info"])
    return df


def load_and_clean_jinetes_hist() -> "pd.DataFrame":
    raw = load_csv(RAW_FILES["jinetes_hist"])
    df  = clean_jinetes_hist(raw)
    save_parquet(df, INTERIM_FILES["jinetes_hist"])
    return df


def load_and_clean_preparadores_info() -> "pd.DataFrame":
    raw = load_csv(RAW_FILES["preparadores_info"])
    df  = clean_preparadores_info(raw)
    save_parquet(df, INTERIM_FILES["preparadores_info"])
    return df


def load_and_clean_preparadores_hist() -> "pd.DataFrame":
    raw = load_csv(RAW_FILES["preparadores_hist"])
    df  = clean_preparadores_hist(raw)
    save_parquet(df, INTERIM_FILES["preparadores_hist"])
    return df


def load_and_clean_genealogy() -> "pd.DataFrame":
    raw = load_excel(RAW_FILES["genealogy"])
    df  = clean_genealogy(raw)
    save_parquet(df, INTERIM_FILES["genealogy"])
    return df


# ── Audit helpers ─────────────────────────────────────────────────────────────

def audit_all(cleaned: dict) -> None:
    """Run full quality report and data dictionary, save to data/reports/."""
    logger.info("Running full quality audit …")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = full_quality_report(cleaned, REPORTS_DIR)

    # Print summary to console
    for name, checks in report.items():
        nr  = checks["nulls"]
        dr  = checks["duplicates"]
        iv  = checks["impossible"]
        high_null = nr[nr["pct_null"] > 10][["column", "pct_null"]].to_dict("records")
        logger.info(
            "[%s] rows=%d  avg_null=%.1f%%  duplicates=%d  impossible_issues=%d  "
            "high_null_cols=%s",
            name,
            dr["n_rows"],
            nr["pct_null"].mean(),
            dr["n_duplicates"],
            len(iv),
            high_null,
        )

    # Data dictionary
    generate_data_dictionary(cleaned, REPORTS_DIR / "data_dictionary.csv")

    logger.info("Audit complete. Reports in: %s", REPORTS_DIR)


def audit_caballos_hist_columns(df: "pd.DataFrame") -> None:
    """Targeted audit of the caballos_historial column semantics."""
    import pandas as pd

    logger.info("=== caballos_hist column audit ===")
    logger.info("pos value_counts (top 20):\n%s", df["pos"].value_counts().head(20))
    logger.info("casilla value_counts (top 20):\n%s", df["casilla"].value_counts().head(20))
    logger.info("mant value_counts (top 20):\n%s", df["mant"].value_counts().head(20))

    # Cross-check: for P.Akupe Taberna on 02-04-2025 compare casilla vs pos
    sample_race = df[
        (df["carrera_nombre"] == "P.Akupe Taberna") &
        (df["fecha"].astype(str).str.startswith("2025-04-02"))
    ][["caballo_id", "mant", "casilla", "pos", "jinete_nombre", "preparador_nombre"]]
    if not sample_race.empty:
        logger.info("Sample race cross-check (P.Akupe Taberna 02-04-2025):\n%s", sample_race.to_string())

    # DNF analysis
    n_dnf = (df["pos"] == 0).sum()
    n_valid = (df["pos"] > 0).sum()
    logger.info("pos=0 (DNF/retired) count: %d (%.1f%%)", n_dnf, 100 * n_dnf / len(df))
    logger.info("pos>0 (valid finish)  count: %d (%.1f%%)", n_valid, 100 * n_valid / len(df))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("STAGE 00 – AUDIT & CLEAN")
    logger.info("=" * 60)

    logger.info("Loading and cleaning source files …")
    jornadas          = load_and_clean_jornadas()
    carreras          = load_and_clean_carreras()
    caballos_hist     = load_and_clean_caballos_hist()
    jinetes_info      = load_and_clean_jinetes_info()
    jinetes_hist      = load_and_clean_jinetes_hist()
    preparadores_info = load_and_clean_preparadores_info()
    preparadores_hist = load_and_clean_preparadores_hist()
    genealogy         = load_and_clean_genealogy()

    audit_caballos_hist_columns(caballos_hist)

    cleaned = {
        "jornadas"         : jornadas,
        "carreras"         : carreras,
        "caballos_hist"    : caballos_hist,
        "jinetes_info"     : jinetes_info,
        "jinetes_hist"     : jinetes_hist,
        "preparadores_info": preparadores_info,
        "preparadores_hist": preparadores_hist,
        "genealogy"        : genealogy,
    }

    audit_all(cleaned)

    logger.info("Stage 00 complete. Interim parquets written to: %s", INTERIM_FILES["jornadas"].parent)


if __name__ == "__main__":
    main()

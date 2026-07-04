"""
Stage 01 – Validate Relations & Reconstruct Participantes
==========================================================
Loads cleaned interim parquets, validates all FK relationships,
builds name→ID lookup maps, and reconstructs the participantes table
(one row per horse per race) with carrera_id, jinete_id, preparador_id.

Run:
    python pipelines/01_validate_relations.py
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    INTERIM_FILES,
    LOG_FORMAT,
    LOG_LEVEL,
    REPORTS_DIR,
)
from bucephalus.io_utils import load_parquet, save_parquet
from bucephalus.validation import (
    build_jinete_name_id_map,
    build_preparador_name_id_map,
    build_validation_report,
    check_fk,
    reconstruct_participantes,
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger("01_validate_relations")


def _print_fk_result(r: dict) -> None:
    status = "OK" if r["n_orphan_values"] == 0 else "WARN"
    logger.info(
        "[%s] %s → orphan_values=%d orphan_rows=%d",
        status, r["check"], r["n_orphan_values"], r["n_orphan_rows"],
    )


def main() -> None:
    logger.info("=" * 60)
    logger.info("STAGE 01 – VALIDATE RELATIONS & RECONSTRUCT PARTICIPANTES")
    logger.info("=" * 60)

    # ── Load interim data ─────────────────────────────────────────────────────
    logger.info("Loading interim parquets …")
    jornadas          = load_parquet(INTERIM_FILES["jornadas"])
    carreras          = load_parquet(INTERIM_FILES["carreras"])
    caballos_hist     = load_parquet(INTERIM_FILES["caballos_hist"])
    jinetes_info      = load_parquet(INTERIM_FILES["jinetes_info"])
    jinetes_hist      = load_parquet(INTERIM_FILES["jinetes_hist"])
    preparadores_info = load_parquet(INTERIM_FILES["preparadores_info"])
    preparadores_hist = load_parquet(INTERIM_FILES["preparadores_hist"])
    genealogy         = load_parquet(INTERIM_FILES["genealogy"])

    # ── FK checks ────────────────────────────────────────────────────────────
    logger.info("Running FK checks …")
    fk_checks = [
        check_fk(carreras, jornadas, "jornada_id", "jornada_id", "carreras", "jornadas"),
        check_fk(jinetes_hist,      jinetes_info,      "jinete_id",     "jinete_id",     "jinetes_hist",  "jinetes_info"),
        check_fk(preparadores_hist, preparadores_info, "preparador_id", "preparador_id", "preps_hist",    "preps_info"),
        check_fk(genealogy,         caballos_hist,     "caballo_id",    "caballo_id",    "genealogy",     "caballos_hist"),
    ]
    for r in fk_checks:
        _print_fk_result(r)

    # ── Jornada / carrera date coverage ──────────────────────────────────────
    jornadas_in_hist = set(
        caballos_hist[["fecha", "hipodromo"]].drop_duplicates()
        .apply(tuple, axis=1)
    )
    jornadas_in_table = set(
        jornadas[["fecha", "hipodromo"]].drop_duplicates()
        .apply(tuple, axis=1)
    )
    hist_not_in_jornadas = jornadas_in_hist - jornadas_in_table
    jornadas_not_in_hist = jornadas_in_table - jornadas_in_hist
    logger.info(
        "Historial (fecha,hipodromo) pairs: %d in hist, %d in jornadas table",
        len(jornadas_in_hist), len(jornadas_in_table),
    )
    logger.info(
        "  → %d hist pairs NOT in jornadas (future or missing jornadas)",
        len(hist_not_in_jornadas),
    )
    logger.info(
        "  → %d jornadas NOT in hist (empty or no horses scraped)",
        len(jornadas_not_in_hist),
    )

    # ── Name → ID lookup maps ─────────────────────────────────────────────────
    logger.info("Building name→ID lookup maps …")
    jinete_map    = build_jinete_name_id_map(caballos_hist, jinetes_hist, genealogy)
    preparador_map= build_preparador_name_id_map(caballos_hist, preparadores_hist, genealogy)

    # Save maps for inspection
    jinete_map.to_csv(REPORTS_DIR / "jinete_name_id_map.csv", index=False)
    preparador_map.to_csv(REPORTS_DIR / "preparador_name_id_map.csv", index=False)
    logger.info("Name maps saved to reports/")

    # Coverage of name resolution
    all_jockey_names = caballos_hist["jinete_nombre"].dropna().unique()
    resolved = set(jinete_map["jinete_nombre"])
    unresolved_j = [n for n in all_jockey_names if n not in resolved]
    logger.info(
        "Jockey name resolution: %d / %d names resolved (%.1f%%) | %d unresolved",
        len(resolved), len(all_jockey_names),
        100 * len(resolved) / max(len(all_jockey_names), 1),
        len(unresolved_j),
    )
    if unresolved_j:
        logger.info("  Sample unresolved jockeys: %s", unresolved_j[:10])

    all_prep_names = caballos_hist["preparador_nombre"].dropna().unique()
    resolved_p = set(preparador_map["preparador_nombre"])
    unresolved_p = [n for n in all_prep_names if n not in resolved_p]
    logger.info(
        "Trainer name resolution: %d / %d names resolved (%.1f%%) | %d unresolved",
        len(resolved_p), len(all_prep_names),
        100 * len(resolved_p) / max(len(all_prep_names), 1),
        len(unresolved_p),
    )

    # Save unresolved for manual review
    if unresolved_j:
        import pandas as pd
        pd.Series(unresolved_j, name="unresolved_jinete_nombre").to_csv(
            REPORTS_DIR / "unresolved_jockey_names.csv", index=False
        )
    if unresolved_p:
        import pandas as pd
        pd.Series(unresolved_p, name="unresolved_preparador_nombre").to_csv(
            REPORTS_DIR / "unresolved_trainer_names.csv", index=False
        )

    # ── Reconstruct participantes ─────────────────────────────────────────────
    logger.info("Reconstructing participantes table …")
    participantes = reconstruct_participantes(
        caballos_hist, jornadas, carreras, jinete_map, preparador_map
    )

    # Save reconstructed participantes
    save_parquet(participantes, INTERIM_FILES["participantes"])

    # Save ID maps in a single parquet for downstream use
    import pandas as pd
    id_maps = pd.concat([
        jinete_map.assign(entity="jinete"),
        preparador_map.rename(columns={"preparador_nombre": "nombre", "preparador_id": "id"})
                      .assign(entity="preparador")
                      .rename(columns={"nombre": "jinete_nombre", "id": "jinete_id"}),
    ], ignore_index=True)
    # (save each map separately as CSV already done)

    # ── Full validation report ────────────────────────────────────────────────
    logger.info("Building full validation report …")
    val_report = build_validation_report(
        jornadas, carreras, caballos_hist,
        jinetes_info, jinetes_hist,
        preparadores_info, preparadores_hist,
        genealogy, participantes,
    )

    # Print coverage section
    cov = val_report["participantes_coverage"]
    logger.info(
        "Participantes coverage → total=%d | carrera_id=%.1f%% | "
        "jinete_id=%.1f%% | preparador_id=%.1f%% | genealogy=%.1f%%",
        cov["total_rows"],
        cov["carrera_id_pct"],
        cov["jinete_id_pct"],
        cov["preparador_id_pct"],
        cov["genealogy_coverage_pct"],
    )

    # Print null audit
    logger.info("Participantes null audit: %s", val_report["participantes_nulls"])

    # Print count discrepancies summary
    disc = val_report["participant_count_discrepancies"]
    n_disc = len(disc) if hasattr(disc, "__len__") else 0
    logger.info(
        "Participant count discrepancies: %d races (see reports/discrepancies.csv)",
        n_disc,
    )
    if n_disc > 0:
        disc.to_csv(REPORTS_DIR / "participant_count_discrepancies.csv", index=False)

    # Save JSON summary of non-dataframe parts of the report
    json_summary = {
        k: v for k, v in val_report.items()
        if not hasattr(v, "to_csv")
    }
    (REPORTS_DIR / "validation_report.json").write_text(
        json.dumps(json_summary, indent=2, default=str)
    )

    logger.info("Stage 01 complete. Participantes: %s", INTERIM_FILES["participantes"])


if __name__ == "__main__":
    main()

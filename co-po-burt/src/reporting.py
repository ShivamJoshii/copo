from pathlib import Path



def write_outputs(results: dict, outdir: Path) -> None:

    results["co_attainment_used"].to_csv(outdir / "co_attainment_used.csv", index=False)

    results["co_report"].to_csv(outdir / "co_report_with_levels.csv", index=False)

    results["merged_detail"].to_csv(outdir / "detail_joined_co_mapping.csv", index=False)

    results["po_long"].to_csv(outdir / "po_pso_attainment_long.csv", index=False)



    results["po_matrix_value"].to_csv(outdir / "po_pso_matrix_value.csv", index=False)

    results["po_matrix_pct"].to_csv(outdir / "po_pso_matrix_percent.csv", index=False)

    results["po_matrix_scale"].to_csv(outdir / "po_pso_matrix_scale_of_3.csv", index=False)

    results["po_matrix_target"].to_csv(outdir / "po_pso_matrix_target_YN.csv", index=False)

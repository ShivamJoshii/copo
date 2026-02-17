import argparse

from pathlib import Path



from src.io_utils import load_co_attainment, load_mapping, load_thresholds, load_targets, load_student_co_scores

from src.nba_math import compute_po_attainment_nba

from src.burt import compute_burt_adjustments_from_students

from src.reporting import write_outputs





def main():

    p = argparse.ArgumentParser()

    p.add_argument("--co_attainment", type=str, required=True, help="CSV: year,course,co,attainment_type,value")

    p.add_argument("--mapping", type=str, required=True, help="CSV: course,co,outcome,weight (0-3)")

    p.add_argument("--thresholds", type=str, required=True, help="CSV: level,min_pct")

    p.add_argument("--targets", type=str, required=True, help="CSV: metric,value")

    p.add_argument("--attainment_type", type=str, default="FINAL", help="Which attainment_type to compute PO/PSO from")

    p.add_argument("--year", type=int, default=None, help="If set, filter to one year")

    p.add_argument("--course", type=str, default=None, help="If set, filter to one course")

    p.add_argument("--mode", choices=["nba", "burt_adjust"], default="nba",

                   help="nba: exact sheet math. burt_adjust: adjusts weights using Burt from student CO data.")

    p.add_argument("--student_co_scores", type=str, default=None,

                   help="Required for burt_adjust. CSV: year,course,student_id,co,co_pct")

    p.add_argument("--outdir", type=str, default="out")

    args = p.parse_args()



    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)



    thresholds = load_thresholds(args.thresholds)

    targets = load_targets(args.targets)



    co_df = load_co_attainment(args.co_attainment)

    map_df = load_mapping(args.mapping)



    # Filter

    if args.year is not None:

        co_df = co_df[co_df["year"] == args.year]

    if args.course is not None:

        co_df = co_df[co_df["course"] == args.course]

        map_df = map_df[map_df["course"] == args.course]



    # Compute optional Burt adjustments

    assoc_df = None

    if args.mode == "burt_adjust":

        if not args.student_co_scores:

            raise ValueError("burt_adjust mode requires --student_co_scores")

        stu_df = load_student_co_scores(args.student_co_scores)

        if args.year is not None:

            stu_df = stu_df[stu_df["year"] == args.year]

        if args.course is not None:

            stu_df = stu_df[stu_df["course"] == args.course]

        assoc_df = compute_burt_adjustments_from_students(stu_df, thresholds)



    results = compute_po_attainment_nba(

        co_attainment=co_df,

        mapping=map_df,

        thresholds=thresholds,

        targets=targets,

        attainment_type=args.attainment_type,

        assoc=assoc_df,  # None in nba mode

    )



    write_outputs(results, outdir)

    print(f"✅ Done. Outputs written to: {outdir.resolve()}")





if __name__ == "__main__":

    main()

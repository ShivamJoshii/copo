import numpy as np

import pandas as pd





def compute_confidence(values, k=1.0, eps=1e-6):

    """

    values: list/array of attainment contributions (CO-level or student-level)

    returns: confidence score in (0, 1]

    """

    values = np.array(values)



    if len(values) == 0:

        return 0.0  # no data → no confidence



    mean = np.mean(values)

    std = np.std(values)



    cv = std / (mean + eps)  # coefficient of variation

    confidence = np.exp(-k * cv)



    return float(np.clip(confidence, 0.0, 1.0))



def pct_to_level(x: float, thresholds: dict) -> int:

    if x >= thresholds[3]:

        return 3

    if x >= thresholds[2]:

        return 2

    if x >= thresholds[1]:

        return 1

    return 0





def compute_burt_adjustments_from_students(student_co_scores: pd.DataFrame, thresholds: dict) -> pd.DataFrame:

    """

    BURT now outputs confidence, not correction.

    Computes confidence scores per (course, co) based on student attainment values.

    Returns columns: course, co, assoc (confidence scores in (0, 1])

    """



    df = student_co_scores.copy()



    # Group by (course, co) and compute confidence from student attainment values (co_pct)

    grp = df.groupby(["course", "co"], as_index=False).agg(

        attainment_values=("co_pct", lambda x: x.tolist())

    )



    # Compute confidence for each group

    grp["assoc"] = grp["attainment_values"].apply(compute_confidence)



    # Return only course, co, assoc

    return grp[["course", "co", "assoc"]]

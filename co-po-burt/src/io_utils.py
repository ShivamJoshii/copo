import pandas as pd





def load_co_attainment(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    required = {"year", "course", "co", "attainment_type", "value"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"co_attainment missing columns: {missing}")

    df["co"] = df["co"].astype(str).str.upper().str.strip()

    df["course"] = df["course"].astype(str).str.strip()

    df["attainment_type"] = df["attainment_type"].astype(str).str.upper().str.strip()

    return df





def load_mapping(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    required = {"course", "co", "outcome", "weight"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"mapping missing columns: {missing}")

    df["co"] = df["co"].astype(str).str.upper().str.strip()

    df["outcome"] = df["outcome"].astype(str).str.upper().str.strip()

    df["course"] = df["course"].astype(str).str.strip()

    df["weight"] = df["weight"].astype(float)

    return df





def load_thresholds(path: str) -> dict:

    """

    returns dict like {3:0.70, 2:0.60, 1:0.50}

    """

    df = pd.read_csv(path)

    required = {"level", "min_pct"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"thresholds missing columns: {missing}")

    out = {int(r["level"]): float(r["min_pct"]) for _, r in df.iterrows()}

    # Basic sanity: must include 1,2,3

    for k in (1, 2, 3):

        if k not in out:

            raise ValueError("thresholds must include levels 1,2,3")

    return out





def load_targets(path: str) -> dict:

    df = pd.read_csv(path)

    required = {"metric", "value"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"targets missing columns: {missing}")

    out = {str(r["metric"]).strip(): float(r["value"]) for _, r in df.iterrows()}

    if "target_level" not in out:

        out["target_level"] = 1.4

    if "scale_max" not in out:

        out["scale_max"] = 3.0

    return out





def load_student_co_scores(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    required = {"year", "course", "student_id", "co", "co_pct"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"student_co_scores missing columns: {missing}")

    df["co"] = df["co"].astype(str).str.upper().str.strip()

    df["course"] = df["course"].astype(str).str.strip()

    df["student_id"] = df["student_id"].astype(str).str.strip()

    df["co_pct"] = df["co_pct"].astype(float)

    return df

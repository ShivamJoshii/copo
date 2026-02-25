import pandas as pd
import numpy as np
import spacy
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Preprocessing Configuration ----------

# Generic academic verbs that don't distinguish between specific learning outcomes
# Removing these helps eliminate false positives caused by common action words
GENERIC_VERBS = {
    "apply", "use", "understand", "demonstrate", "analyze", "evaluate",
    "identify", "explain", "describe", "perform", "develop", "design",
    "implement", "assess", "measure"
}

# Department-specific blacklists for fine-tuning
# Can be extended based on specific institutional needs
DEPT_BLACKLIST = {
    "engineering": {"apply", "use", "demonstrate"},
    "business": {"analyze", "evaluate", "assess"},
    "cs": {"implement", "design"}
}

@st.cache_resource
def _load_spacy():
    """Load spacy model once and cache it."""
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy model 'en_core_web_sm' loaded successfully")
        return nlp
    except OSError:
        # Fallback to blank model with sentencizer instead of downloading at runtime
        print("⚠ Warning: spaCy model 'en_core_web_sm' not found. Using fallback mode.")
        print("  Lemmatization and stopword removal may be reduced in quality.")
        print("  Install the model at build time: python -m spacy download en_core_web_sm")
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp


def preprocess_light(text: str) -> str:
    """
    Light preprocessing: remove only stopwords and punctuation.

    Used as fallback when aggressive preprocessing leaves too few tokens.

    Args:
        text: Input text to preprocess

    Returns:
        Lightly cleaned text
    """
    nlp = _load_spacy()
    doc = nlp(text.lower())

    tokens = []
    for token in doc:
        # Skip stopwords (a, the, is, etc.)
        # In fallback mode (blank model), is_stop may not work reliably
        if hasattr(token, 'is_stop') and token.is_stop:
            continue

        # Skip punctuation and symbols
        if token.pos_ in {"PUNCT", "SYM", "SPACE"}:
            continue

        # Keep all other meaningful tokens
        if token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}:
            # Use lemma_ if available (full model), else use lowercased text (fallback)
            tokens.append(token.lemma_ if token.lemma_ != token.text else token.text.lower())

    return " ".join(tokens)


def preprocess_minimal(text: str) -> str:
    """
    Minimal preprocessing: clean whitespace only, preserve full semantic meaning.

    This is the "human-like" mode - keeps all words including verbs,
    stopwords, everything. The sentence transformer model understands
    context better when given complete sentences.

    Args:
        text: Input text to preprocess

    Returns:
        Cleaned text with full semantic content preserved
    """
    # Just normalize whitespace and lowercase
    text = text.lower().strip()
    # Collapse multiple spaces/newlines into single space
    import re
    text = re.sub(r'\s+', ' ', text)
    return text


def preprocess_aggressive(text: str, dept: str = "general") -> str:
    """
    Aggressive preprocessing: remove stopwords, generic verbs, and punctuation.

    This reduces false positives by focusing on domain-specific technical content
    rather than generic academic language.

    Args:
        text: Input text to preprocess
        dept: Department identifier for specialized filtering (engineering, business, cs, or general)

    Returns:
        Aggressively cleaned text with stopwords and generic verbs removed
    """
    nlp = _load_spacy()

    # Combine generic verbs with department-specific blacklist
    blacklist = GENERIC_VERBS | DEPT_BLACKLIST.get(dept, set())
    doc = nlp(text.lower())

    tokens = []
    for token in doc:
        # Skip stopwords (a, the, is, etc.)
        # In fallback mode (blank model), is_stop may not work reliably
        if hasattr(token, 'is_stop') and token.is_stop:
            continue

        # Use lemma_ if available (full model), else use lowercased text (fallback)
        lemma = token.lemma_ if token.lemma_ != token.text else token.text.lower()

        # Filter out generic academic verbs that don't add semantic value
        if token.pos_ == "VERB" and lemma in blacklist:
            continue

        # Skip punctuation and symbols
        if token.pos_ in {"PUNCT", "SYM", "SPACE"}:
            continue

        # Keep nouns, proper nouns, adjectives, and technical verbs
        # These carry the semantic content that distinguishes CO/PO statements
        if token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}:
            tokens.append(lemma)

    return " ".join(tokens)


def preprocess(text: str, dept: str = "general", mode: str = "aggressive_with_fallback") -> str:
    """
    Preprocess text with configurable modes to balance precision and recall.

    This reduces false positives by focusing on domain-specific technical content
    rather than generic academic language, while avoiding over-aggressive cleaning
    that can destroy meaning.

    Args:
        text: Input text to preprocess
        dept: Department identifier for specialized filtering (engineering, business, cs, or general)
        mode: Preprocessing mode - "minimal", "aggressive", "light", or "aggressive_with_fallback"

    Returns:
        Cleaned text according to the specified mode
    """
    if mode == "minimal":
        return preprocess_minimal(text)

    if mode == "light":
        return preprocess_light(text)

    if mode == "aggressive":
        return preprocess_aggressive(text, dept)

    # Default: aggressive_with_fallback
    aggressive_result = preprocess_aggressive(text, dept)

    # If aggressive preprocessing left too few tokens, fall back to light
    if len(aggressive_result.split()) < 3:
        return preprocess_light(text)

    return aggressive_result


# ---------- Sentence Embedding Model ----------

@st.cache_resource
def _load_sentence_model():
    """Load sentence-transformers model once and cache it."""
    # all-MiniLM-L6-v2 is lightweight and performs well for semantic similarity
    return SentenceTransformer("all-MiniLM-L6-v2")


def detect_id_column(df, keywords):

    """

    Detect column containing IDs like CO1, PO1, PSO1

    """

    # 1) exact matches first (most reliable)

    preferred = []

    for k in keywords:

        preferred += [k, f"{k}_id", f"{k}id", f"{k} no", f"{k}_no"]

    preferred = {p.lower() for p in preferred}



    for col in df.columns:

        if col.lower().strip() in preferred:

            return col



    # 2) fallback: substring match

    for col in df.columns:

        col_l = col.lower()

        if any(k in col_l for k in keywords):

            return col



    raise ValueError(f"Could not detect ID column. Columns found: {list(df.columns)}")





def detect_text_column(df, id_col):

    """

    Detect the text/statement column (anything except ID)

    """

    candidates = [c for c in df.columns if c != id_col]



    if not candidates:

        raise ValueError("Could not detect text column")



    # Drop columns that are obviously not text

    blacklist = {"weight", "similarity", "attainment", "score", "marks", "percentage"}

    candidates2 = []

    for c in candidates:

        if c.lower().strip() not in blacklist:

            candidates2.append(c)

    candidates = candidates2 or candidates



    # Choose column with greatest average string length

    best_col = None

    best_len = -1



    for c in candidates:

        s = df[c].dropna().astype(str).str.strip()

        if len(s) == 0:

            continue

        avg_len = s.str.len().mean()

        if avg_len > best_len:

            best_len = avg_len

            best_col = c



    if best_col is None:

        raise ValueError(f"Could not detect text column. Candidates were: {candidates}")



    return best_col





def similarity_to_weight_bins(sim: float) -> int:
    """
    Convert similarity score to weight (0-3) using fixed bins.

    Fixed bins:
        0.00-0.25 => 0
        0.25-0.50 => 1
        0.50-0.75 => 2
        0.75-1.00 => 3

    Args:
        sim: Similarity score in [0, 1]

    Returns:
        Weight in {0, 1, 2, 3}
    """
    if sim < 0.25:
        return 0
    elif sim < 0.50:
        return 1
    elif sim < 0.75:
        return 2
    else:
        return 3


def similarity_to_weight(sim, t3=0.75, t2=0.50, t1=0.25, min_threshold=0.0):
    """
    Convert similarity score to weight (0-3) based on thresholds.

    DEPRECATED: Use similarity_to_weight_bins() for new code.
    This function is kept for backwards compatibility with existing CO-PO mapping.

    Args:
        sim: Similarity score in [0, 1]
        t3: Threshold for weight 3 (default 0.75)
        t2: Threshold for weight 2 (default 0.50)
        t1: Threshold for weight 1 (default 0.25)
        min_threshold: Minimum similarity for any non-zero weight (default 0.0)

    Returns:
        Weight in {0, 1, 2, 3}
    """
    # If below minimum threshold, always return 0
    if sim < min_threshold:
        return 0

    if sim >= t3:
        return 3
    if sim >= t2:
        return 2
    if sim >= t1:
        return 1
    return 0





# ---------- Legacy BERT code removed ----------
# Now using sentence-transformers for better performance and simpler API





def generate_co_po_mapping(co_df: pd.DataFrame, po_df: pd.DataFrame,
                          threshold: float = 0.55, dept: str = "general",
                          preprocess_mode: str = "aggressive_with_fallback") -> pd.DataFrame:
    """
    Generate CO-PO mapping using sentence embeddings and preprocessing.

    Args:
        co_df: DataFrame containing Course Outcomes with ID and text columns
        po_df: DataFrame containing Program Outcomes with ID and text columns
        threshold: Minimum similarity threshold for considering a mapping (default 0.55)
        dept: Department identifier for specialized preprocessing (default "general")
        preprocess_mode: Preprocessing mode - "aggressive", "light", or "aggressive_with_fallback" (default)

    Returns:
        DataFrame with columns: co, outcome, similarity, weight
    """
    # ---- detect columns safely ----
    co_id_col = detect_id_column(co_df, ["co"])
    po_id_col = detect_id_column(po_df, ["po", "pso", "outcome"])

    co_text_col = detect_text_column(co_df, co_id_col)
    po_text_col = detect_text_column(po_df, po_id_col)

    # ---- clean data ----
    co_df = co_df.dropna(subset=[co_text_col])
    po_df = po_df.dropna(subset=[po_text_col])

    co_ids = co_df[co_id_col].astype(str).str.strip().tolist()
    po_ids = po_df[po_id_col].astype(str).str.strip().tolist()

    # normalize whitespace
    co_texts = (
        co_df[co_text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    )
    po_texts = (
        po_df[po_text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    )

    # ---- apply preprocessing to remove stopwords and generic verbs ----
    co_texts_clean = [preprocess(text, dept, preprocess_mode) for text in co_texts]
    po_texts_clean = [preprocess(text, dept, preprocess_mode) for text in po_texts]

    # ---- sentence embeddings (normalized) ----
    model = _load_sentence_model()
    co_emb = model.encode(co_texts_clean, normalize_embeddings=True, show_progress_bar=False)
    po_emb = model.encode(po_texts_clean, normalize_embeddings=True, show_progress_bar=False)

    # ---- compute similarity matrix ----
    # With normalized embeddings, cosine similarity is already in [-1, 1] range
    sim_matrix = cosine_similarity(co_emb, po_emb)

    # Convert cosine similarity from [-1, 1] to [0, 1] scale
    sim_matrix = (sim_matrix + 1.0) / 2.0
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)

    # ---- build mapping ----
    rows = []
    for i, co in enumerate(co_ids):
        for j, outcome in enumerate(po_ids):
            sim01 = float(sim_matrix[i, j])

            # Apply threshold: force weight to 0 if similarity below threshold
            if sim01 < threshold:
                weight = 0
            else:
                weight = similarity_to_weight(sim01, t3=0.75, t2=0.50, t1=0.25)

            rows.append(
                {
                    "co": co,
                    "outcome": outcome,
                    "similarity": round(sim01, 4),
                    "weight": weight,
                }
            )

    return pd.DataFrame(rows)


@st.cache_data
def _compute_embeddings(texts: list, _model):
    """
    Compute embeddings for a list of texts. Cached for performance.

    Args:
        texts: List of text strings to embed
        _model: SentenceTransformer model (underscore prefix prevents caching)

    Returns:
        numpy array of embeddings (normalized)
    """
    return _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def generate_co_to_single_outcome_mapping(
    co_df: pd.DataFrame,
    outcome_id: str,
    outcome_text: str,
    dept: str = "general",
    preprocess_mode: str = "aggressive_with_fallback"
) -> pd.DataFrame:
    """
    Generate mapping for ALL COs against a SINGLE selected PO/PSO.

    Uses normalized embeddings and fixed weight bins for stable, predictable results.

    Fixed weight bins:
        0.00-0.25 => 0
        0.25-0.50 => 1
        0.50-0.75 => 2
        0.75-1.00 => 3

    Args:
        co_df: DataFrame containing Course Outcomes with ID and text columns
        outcome_id: The ID of the selected outcome (e.g., "PO1", "PSO2")
        outcome_text: The text description of the selected outcome
        dept: Department identifier for specialized preprocessing (default "general")
        preprocess_mode: Preprocessing mode - "aggressive", "light", or "aggressive_with_fallback"

    Returns:
        DataFrame with columns: co, co_text, outcome, outcome_text, similarity, weight
        Sorted by similarity (descending)
    """
    # ---- detect columns safely ----
    co_id_col = detect_id_column(co_df, ["co"])
    co_text_col = detect_text_column(co_df, co_id_col)

    # ---- clean data ----
    co_df = co_df.dropna(subset=[co_text_col])

    co_ids = co_df[co_id_col].astype(str).str.strip().tolist()

    # normalize whitespace
    co_texts = (
        co_df[co_text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    )

    # ---- apply preprocessing to remove stopwords and generic verbs ----
    co_texts_clean = [preprocess(text, dept, preprocess_mode) for text in co_texts]
    outcome_text_clean = preprocess(outcome_text, dept, preprocess_mode)

    # ---- sentence embeddings with caching (normalized) ----
    model = _load_sentence_model()
    co_emb = _compute_embeddings(co_texts_clean, model)
    outcome_emb = _compute_embeddings([outcome_text_clean], model)

    # ---- compute similarity vector ----
    # With normalized embeddings, cosine similarity is already in [0, 1] range
    sim_vector = cosine_similarity(co_emb, outcome_emb).flatten()

    # Clip to ensure [0, 1] range (no conversion needed)
    sim_vector = np.clip(sim_vector, 0.0, 1.0)

    # ---- build mapping ----
    rows = []
    for i, co in enumerate(co_ids):
        sim01 = float(sim_vector[i])

        # Use fixed bins for weight assignment
        weight = similarity_to_weight_bins(sim01)

        rows.append(
            {
                "co": co,
                "co_text": co_texts[i],
                "outcome": outcome_id,
                "outcome_text": outcome_text,
                "similarity": round(sim01, 4),
                "weight": weight,
            }
        )

    # Return sorted by similarity (descending) by default
    df = pd.DataFrame(rows)
    return df.sort_values("similarity", ascending=False).reset_index(drop=True)

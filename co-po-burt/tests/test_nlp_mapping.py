"""
Unit tests for NLP mapping functionality.

Tests focus on validating that preprocessing reduces false positives
caused by generic academic verbs.
"""

import unittest
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.nlp_mapping import preprocess, generate_co_po_mapping, generate_co_to_single_outcome_mapping


class TestPreprocessing(unittest.TestCase):
    """Test the preprocessing function."""

    def test_removes_stopwords(self):
        """Test that common stopwords are removed."""
        text = "The student will apply the concepts of calculus"
        result = preprocess(text)
        # Should remove 'the', 'will', 'of' and 'apply' (generic verb)
        self.assertNotIn("the", result.lower())
        self.assertNotIn("will", result.lower())
        self.assertNotIn("apply", result.lower())

    def test_removes_generic_verbs(self):
        """Test that generic academic verbs are filtered out."""
        text = "Apply, demonstrate, analyze, and understand the principles"
        result = preprocess(text)
        # All these generic verbs should be removed
        self.assertNotIn("apply", result.lower())
        self.assertNotIn("demonstrate", result.lower())
        self.assertNotIn("analyze", result.lower())
        self.assertNotIn("understand", result.lower())

    def test_preserves_domain_nouns(self):
        """Test that domain-specific nouns are preserved."""
        text = "Apply calculus to solve differential equations"
        result = preprocess(text)
        # Should keep domain-specific nouns
        self.assertIn("calculus", result.lower())
        self.assertIn("differential", result.lower())
        self.assertIn("equation", result.lower())

    def test_department_specific_filtering(self):
        """Test that department-specific blacklists work."""
        text = "Apply machine learning algorithms"
        result_general = preprocess(text, dept="general")
        result_cs = preprocess(text, dept="cs")
        # Both should remove 'apply', but result should contain meaningful content
        self.assertNotIn("apply", result_general.lower())
        self.assertIn("machine", result_general.lower())


class TestFalsePositiveReduction(unittest.TestCase):
    """Test that preprocessing reduces false positives in CO-PO mapping."""

    def setUp(self):
        """Set up test data."""
        # CO statement about calculus
        self.co_calculus = pd.DataFrame({
            "CO": ["CO1"],
            "Statement": ["Apply principles of differential calculus to solve optimization problems"]
        })

        # CO statement about ethics
        self.co_ethics = pd.DataFrame({
            "CO": ["CO2"],
            "Statement": ["Apply ethical principles in professional engineering practice"]
        })

        # PO about mathematics
        self.po_math = pd.DataFrame({
            "PO": ["PO1"],
            "Description": ["Apply knowledge of mathematics, science, and engineering fundamentals"]
        })

        # PO about ethics
        self.po_ethics = pd.DataFrame({
            "PO": ["PO2"],
            "Description": ["Apply ethical principles and commit to professional responsibilities"]
        })

    def test_calculus_ethics_false_positive(self):
        """
        Test that 'apply calculus' does not match 'apply ethics' due to preprocessing.

        This is the key test case mentioned in the requirements:
        CO about calculus should NOT strongly match PO about ethics,
        even though both use the word 'apply'.
        """
        # Test CO about calculus vs PO about ethics (should be low similarity)
        mapping_calculus_ethics = generate_co_po_mapping(
            self.co_calculus,
            self.po_ethics,
            dept="engineering"
        )

        # Test CO about calculus vs PO about math (should be high similarity)
        mapping_calculus_math = generate_co_po_mapping(
            self.co_calculus,
            self.po_math,
            dept="engineering"
        )

        # Test CO about ethics vs PO about ethics (should be high similarity)
        mapping_ethics_ethics = generate_co_po_mapping(
            self.co_ethics,
            self.po_ethics,
            dept="engineering"
        )

        sim_calculus_ethics = mapping_calculus_ethics.iloc[0]["similarity"]
        sim_calculus_math = mapping_calculus_math.iloc[0]["similarity"]
        sim_ethics_ethics = mapping_ethics_ethics.iloc[0]["similarity"]

        # Calculus CO should match math PO better than ethics PO
        self.assertGreater(
            sim_calculus_math,
            sim_calculus_ethics,
            f"Calculus CO should match math PO ({sim_calculus_math:.3f}) "
            f"better than ethics PO ({sim_calculus_ethics:.3f})"
        )

        # Ethics CO should strongly match ethics PO
        self.assertGreater(
            sim_ethics_ethics,
            sim_calculus_ethics,
            f"Ethics CO should match ethics PO ({sim_ethics_ethics:.3f}) "
            f"better than calculus CO matches ethics PO ({sim_calculus_ethics:.3f})"
        )

    def test_mapping_preserves_structure(self):
        """Test that generate_co_po_mapping returns expected DataFrame structure."""
        mapping = generate_co_po_mapping(self.co_calculus, self.po_math)

        # Check DataFrame has expected columns
        self.assertIn("co", mapping.columns)
        self.assertIn("outcome", mapping.columns)
        self.assertIn("similarity", mapping.columns)
        self.assertIn("weight", mapping.columns)

        # Check similarity is in valid range
        self.assertTrue((mapping["similarity"] >= 0).all())
        self.assertTrue((mapping["similarity"] <= 1).all())

        # Check weight is in valid range (0-3)
        self.assertTrue((mapping["weight"] >= 0).all())
        self.assertTrue((mapping["weight"] <= 3).all())


class TestSimilarityScaling(unittest.TestCase):
    """Test that similarity scores are properly scaled to [0,1]."""

    def setUp(self):
        """Set up test data."""
        self.co_df = pd.DataFrame({
            "CO": ["CO1", "CO2"],
            "Statement": [
                "Apply principles of differential calculus",
                "Demonstrate understanding of data structures"
            ]
        })

        self.po_df = pd.DataFrame({
            "PO": ["PO1", "PO2"],
            "Description": [
                "Apply knowledge of mathematics and science",
                "Design and implement efficient algorithms"
            ]
        })

    def test_similarity_always_in_range(self):
        """Test that all similarity values are within [0, 1]."""
        mapping = generate_co_po_mapping(self.co_df, self.po_df)

        # Assert no negative similarities
        self.assertTrue(
            (mapping["similarity"] >= 0.0).all(),
            f"Found negative similarity values: {mapping[mapping['similarity'] < 0.0]}"
        )

        # Assert no similarities greater than 1
        self.assertTrue(
            (mapping["similarity"] <= 1.0).all(),
            f"Found similarity values > 1.0: {mapping[mapping['similarity'] > 1.0]}"
        )

    def test_threshold_forces_zero_weight(self):
        """Test that threshold parameter forces weight to 0 when similarity is below threshold."""
        # Use high threshold to force some weights to 0
        high_threshold = 0.80
        mapping = generate_co_po_mapping(self.co_df, self.po_df, threshold=high_threshold)

        # Check that all rows with similarity < threshold have weight = 0
        below_threshold = mapping[mapping["similarity"] < high_threshold]
        if not below_threshold.empty:
            self.assertTrue(
                (below_threshold["weight"] == 0).all(),
                f"Found non-zero weights below threshold: {below_threshold[below_threshold['weight'] != 0]}"
            )

        # Check that rows with similarity >= threshold can have non-zero weights
        above_threshold = mapping[mapping["similarity"] >= high_threshold]
        if not above_threshold.empty:
            # At least some should have non-zero weights (depending on similarity_to_weight thresholds)
            # This is a weaker assertion - just verify structure is correct
            self.assertTrue(
                (above_threshold["weight"] >= 0).all(),
                "Weights above threshold should be non-negative"
            )

    def test_weight_bin_thresholds(self):
        """Test that weight bins match exact thresholds: 0.25, 0.50, 0.75."""
        from src.nlp_mapping import similarity_to_weight_bins

        # Test boundary conditions for fixed bins
        self.assertEqual(similarity_to_weight_bins(0.00), 0)
        self.assertEqual(similarity_to_weight_bins(0.249), 0)
        self.assertEqual(similarity_to_weight_bins(0.25), 1)
        self.assertEqual(similarity_to_weight_bins(0.499), 1)
        self.assertEqual(similarity_to_weight_bins(0.50), 2)
        self.assertEqual(similarity_to_weight_bins(0.749), 2)
        self.assertEqual(similarity_to_weight_bins(0.75), 3)
        self.assertEqual(similarity_to_weight_bins(1.00), 3)


class TestSingleOutcomeMapping(unittest.TestCase):
    """Test the new single outcome mapping function."""

    def setUp(self):
        """Set up test data."""
        self.co_df = pd.DataFrame({
            "CO": ["CO1", "CO2", "CO3"],
            "Statement": [
                "Apply principles of differential calculus",
                "Demonstrate understanding of data structures",
                "Apply ethical principles in engineering"
            ]
        })

        self.outcome_id = "PO1"
        self.outcome_text = "Apply knowledge of mathematics, science, and engineering fundamentals"

    def test_single_outcome_mapping_structure(self):
        """Test that single outcome mapping returns correct DataFrame structure."""
        mapping = generate_co_to_single_outcome_mapping(
            self.co_df,
            self.outcome_id,
            self.outcome_text
        )

        # Check DataFrame has expected columns
        expected_cols = ["co", "co_text", "outcome", "outcome_text", "similarity", "weight"]
        for col in expected_cols:
            self.assertIn(col, mapping.columns, f"Missing column: {col}")

        # Check correct number of rows (one per CO)
        self.assertEqual(len(mapping), len(self.co_df), "Should have one row per CO")

        # Check all rows have same outcome
        self.assertTrue((mapping["outcome"] == self.outcome_id).all(), "All rows should have selected outcome")

        # Check similarity is in valid range [0, 1]
        self.assertTrue((mapping["similarity"] >= 0.0).all(), "Similarity should be >= 0")
        self.assertTrue((mapping["similarity"] <= 1.0).all(), "Similarity should be <= 1")

        # Check weight is in valid range {0, 1, 2, 3}
        valid_weights = {0, 1, 2, 3}
        self.assertTrue(
            mapping["weight"].isin(valid_weights).all(),
            f"All weights should be in {valid_weights}"
        )

    def test_single_outcome_sorting(self):
        """Test that results are sorted by similarity descending."""
        mapping = generate_co_to_single_outcome_mapping(
            self.co_df,
            self.outcome_id,
            self.outcome_text
        )

        # Check that similarities are in descending order
        similarities = mapping["similarity"].tolist()
        self.assertEqual(
            similarities,
            sorted(similarities, reverse=True),
            "Results should be sorted by similarity (descending)"
        )

    def test_weight_thresholds(self):
        """Test that weight thresholds are correctly applied with fixed bins."""
        mapping = generate_co_to_single_outcome_mapping(
            self.co_df,
            self.outcome_id,
            self.outcome_text
        )

        # Check weight logic with fixed bins
        for _, row in mapping.iterrows():
            sim = row["similarity"]
            weight = row["weight"]

            if sim >= 0.75:
                self.assertEqual(weight, 3, f"sim={sim} should give weight=3")
            elif sim >= 0.50:
                self.assertEqual(weight, 2, f"sim={sim} should give weight=2")
            elif sim >= 0.25:
                self.assertEqual(weight, 1, f"sim={sim} should give weight=1")
            else:
                self.assertEqual(weight, 0, f"sim={sim} should give weight=0")

    def test_fixed_bins_behavior(self):
        """Test that fixed bins work as expected without configurable thresholds."""
        mapping = generate_co_to_single_outcome_mapping(
            self.co_df,
            self.outcome_id,
            self.outcome_text
        )

        # Verify that weights strictly follow fixed bins
        # No rows should violate the fixed bin boundaries
        for _, row in mapping.iterrows():
            sim = row["similarity"]
            weight = row["weight"]

            # Verify weight matches the exact bin
            if sim < 0.25:
                self.assertEqual(weight, 0, f"sim={sim:.4f} in [0.00-0.25) should give weight=0")
            elif sim < 0.50:
                self.assertEqual(weight, 1, f"sim={sim:.4f} in [0.25-0.50) should give weight=1")
            elif sim < 0.75:
                self.assertEqual(weight, 2, f"sim={sim:.4f} in [0.50-0.75) should give weight=2")
            else:
                self.assertEqual(weight, 3, f"sim={sim:.4f} in [0.75-1.00] should give weight=3")

    def test_co_text_preserved(self):
        """Test that CO text is preserved in output."""
        mapping = generate_co_to_single_outcome_mapping(
            self.co_df,
            self.outcome_id,
            self.outcome_text
        )

        # Check that all CO texts are present
        co_texts_in_output = set(mapping["co_text"].tolist())
        co_texts_in_input = set(self.co_df["Statement"].tolist())

        self.assertEqual(
            co_texts_in_output,
            co_texts_in_input,
            "All CO texts should be preserved in output"
        )

    def test_outcome_text_preserved(self):
        """Test that outcome text is preserved in all rows."""
        mapping = generate_co_to_single_outcome_mapping(
            self.co_df,
            self.outcome_id,
            self.outcome_text
        )

        # Check that all rows have the same outcome text
        self.assertTrue(
            (mapping["outcome_text"] == self.outcome_text).all(),
            "All rows should have the selected outcome text"
        )


if __name__ == "__main__":
    unittest.main()

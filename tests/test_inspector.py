import unittest

import pandas as pd

from dataset_inspector import analyze_dataframe, prepare_cleaned_copy


class AnalyzeDataFrameTests(unittest.TestCase):
    def test_reports_missing_duplicates_and_constant_columns(self):
        frame = pd.DataFrame({
            "salary": [10.0, 20.0, None, 20.0],
            "department": ["A", "B", "A", "B"],
            "source": ["imported"] * 4,
        })
        report = analyze_dataframe(frame)

        self.assertEqual(report.rows, 4)
        self.assertEqual(report.columns, 3)
        self.assertEqual(report.duplicate_rows, 1)
        self.assertEqual(report.missing_cells, 1)
        self.assertAlmostEqual(report.missing_percent, 100 / 12)
        titles = {finding.title for finding in report.findings}
        self.assertIn("salary: 25.0% missing", titles)
        self.assertIn("1 duplicate rows", titles)
        self.assertIn("source has one unique value", titles)
        self.assertLessEqual(report.ml_readiness_score, report.health_score)

    def test_detects_outliers_and_high_correlation(self):
        frame = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 100],
            "x_scaled": [10, 20, 30, 40, 50, 1000],
        })
        report = analyze_dataframe(frame)

        categories = {finding.category for finding in report.findings}
        self.assertIn("Potential outliers", categories)
        self.assertIn("High correlation", categories)

    def test_class_imbalance_requires_an_explicit_target(self):
        frame = pd.DataFrame({"label": ["common"] * 19 + ["rare"], "feature": range(20)})
        without_target = analyze_dataframe(frame)
        with_target = analyze_dataframe(frame, target_column="label")

        self.assertNotIn("Class imbalance", {item.category for item in without_target.findings})
        self.assertIn("Class imbalance", {item.category for item in with_target.findings})

    def test_detects_mixed_numeric_and_text_values(self):
        frame = pd.DataFrame({"amount": ["10", "12", "13", "unknown", "14"]})
        report = analyze_dataframe(frame)

        self.assertIn(
            "Potential invalid values",
            {finding.category for finding in report.findings},
        )

    def test_rejects_empty_datasets_and_unknown_target(self):
        with self.assertRaises(ValueError):
            analyze_dataframe(pd.DataFrame())
        with self.assertRaises(ValueError):
            analyze_dataframe(pd.DataFrame({"x": [1]}), target_column="missing")


class CleanedCopyTests(unittest.TestCase):
    def test_imputes_numeric_and_categorical_and_deduplicates_without_mutating_input(self):
        frame = pd.DataFrame({
            "amount": [10.0, None, 10.0],
            "group": ["A", "B", "A"],
        })
        cleaned = prepare_cleaned_copy(frame)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["amount"].tolist(), [10.0, 10.0])
        self.assertEqual(cleaned["group"].tolist(), ["A", "B"])
        self.assertTrue(frame["amount"].isna().iloc[1])

    def test_cleaning_options_can_be_disabled(self):
        frame = pd.DataFrame({"x": [1.0, None, 1.0]})
        cleaned = prepare_cleaned_copy(frame, impute_missing=False, remove_duplicates=False)

        self.assertEqual(len(cleaned), 3)
        self.assertTrue(cleaned["x"].isna().iloc[1])

    def test_imputation_does_not_create_new_duplicate_removals(self):
        frame = pd.DataFrame({"x": [1.0, 1.0, None]})
        cleaned = prepare_cleaned_copy(frame)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["x"].tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()

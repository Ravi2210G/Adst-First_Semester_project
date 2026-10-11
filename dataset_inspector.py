"""Dataset profiling and conservative preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

Severity = Literal["Critical", "High", "Medium", "Low", "Healthy"]


@dataclass(frozen=True)
class Finding:
    category: str
    severity: Severity
    title: str
    detail: str
    recommendation: str


@dataclass(frozen=True)
class DatasetReport:
    rows: int
    columns: int
    numeric_columns: int
    categorical_columns: int
    datetime_columns: int
    duplicate_rows: int
    missing_cells: int
    missing_percent: float
    health_score: int
    ml_readiness_score: int
    findings: tuple[Finding, ...]
    recommendations: tuple[str, ...]
    column_summary: pd.DataFrame


def _severity_for_percent(percent: float) -> Severity:
    if percent >= 20:
        return "Critical"
    if percent >= 10:
        return "High"
    if percent >= 2:
        return "Medium"
    return "Low"


def _clean_name(value: object) -> str:
    return str(value).strip() or "(unnamed column)"


def analyze_dataframe(
    frame: pd.DataFrame, target_column: str | None = None
) -> DatasetReport:
    """Inspect a dataframe and return quality findings and summary metrics."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("The dataset must contain at least one row and one column.")

    rows, columns = frame.shape
    numeric_names = [
        name for name in frame.columns
        if is_numeric_dtype(frame[name].dtype) and not is_bool_dtype(frame[name].dtype)
    ]
    datetime_names = [
        name for name in frame.columns if is_datetime64_any_dtype(frame[name].dtype)
    ]
    categorical_names = [
        name for name in frame.columns
        if name not in numeric_names and name not in datetime_names
    ]
    duplicate_rows = int(frame.duplicated().sum())
    missing_cells = int(frame.isna().sum().sum())
    missing_percent = missing_cells / (rows * columns) * 100
    findings: list[Finding] = []
    recommendations: list[str] = []

    per_column_missing = frame.isna().sum()
    for name, count in per_column_missing.items():
        if count:
            percent = float(count / rows * 100)
            label = _clean_name(name)
            method = "median" if name in numeric_names else "most frequent value"
            findings.append(Finding(
                "Missing values",
                _severity_for_percent(percent),
                f"{label}: {percent:.1f}% missing",
                f"{int(count):,} of {rows:,} rows are missing a value in this column.",
                f"Consider {method} imputation; investigate whether the missingness is systematic.",
            ))
            recommendations.append(f"Impute missing values in {label} using the {method}.")

    if duplicate_rows:
        duplicate_percent = duplicate_rows / rows * 100
        findings.append(Finding(
            "Duplicate records",
            "High" if duplicate_percent >= 5 else "Medium",
            f"{duplicate_rows:,} duplicate rows",
            f"{duplicate_percent:.1f}% of rows are exact duplicates across all columns.",
            "Review and remove exact duplicates before training to avoid overweighting repeated records.",
        ))
        recommendations.append(f"Review and remove {duplicate_rows:,} exact duplicate rows.")

    constant_names = [
        name for name in frame.columns
        if frame[name].nunique(dropna=False) <= 1
    ]
    for name in constant_names:
        findings.append(Finding(
            "Constant column",
            "Medium",
            f"{_clean_name(name)} has one unique value",
            "A constant feature cannot help distinguish between observations.",
            "Consider dropping this column after confirming it is not required as an identifier.",
        ))
        recommendations.append(f"Review constant column {_clean_name(name)} for removal.")

    invalid_count = 0
    for name in categorical_names:
        series = frame[name].dropna()
        if not len(series):
            continue
        converted = pd.to_numeric(series, errors="coerce")
        numeric_fraction = float(converted.notna().mean())
        invalid_values = series[converted.isna()].drop_duplicates()
        if 0.8 <= numeric_fraction < 1 and len(invalid_values):
            invalid_count += len(invalid_values)
            examples = ", ".join(repr(value) for value in invalid_values.iloc[:3])
            findings.append(Finding(
                "Potential invalid values",
                "Medium",
                f"{_clean_name(name)} mixes numbers and text",
                f"Most values look numeric, but some do not (for example: {examples}).",
                "Check spelling, units, and sentinel strings before converting this column to numeric.",
            ))
            recommendations.append(
                f"Inspect non-numeric entries in {_clean_name(name)} before type conversion."
            )

    outlier_counts: dict[object, int] = {}
    for name in numeric_names:
        values = frame[name].dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((values < lower) | (values > upper)).sum())
        if outlier_count:
            outlier_counts[name] = outlier_count
            percent = outlier_count / len(values) * 100
            findings.append(Finding(
                "Potential outliers",
                "High" if percent >= 5 else "Medium" if percent >= 1 else "Low",
                f"{_clean_name(name)}: {outlier_count:,} IQR outliers",
                f"{percent:.1f}% of non-missing values fall outside [{lower:.3g}, {upper:.3g}].",
                "Validate these values with subject-matter context; investigate before removing or capping them.",
            ))
            recommendations.append(
                f"Investigate {outlier_count:,} potential outliers in {_clean_name(name)}."
            )

    correlation_pairs: list[tuple[object, object, float]] = []
    if len(numeric_names) > 1:
        correlations = frame[numeric_names].corr().abs()
        for left_index, left in enumerate(numeric_names):
            for right in numeric_names[left_index + 1:]:
                correlation = float(correlations.loc[left, right])
                if np.isfinite(correlation) and correlation >= 0.85:
                    correlation_pairs.append((left, right, correlation))
                    findings.append(Finding(
                        "High correlation",
                        "Medium",
                        f"{_clean_name(left)} ↔ {_clean_name(right)}: {correlation:.2f}",
                        "These numeric features move together strongly and may contain redundant information.",
                        "Check for multicollinearity and consider whether both features are needed for your model.",
                    ))
                    recommendations.append(
                        f"Check redundancy between {_clean_name(left)} and {_clean_name(right)} (r={correlation:.2f})."
                    )

    if target_column is not None:
        if target_column not in frame.columns:
            raise ValueError(f"Target column {target_column!r} is not in the dataset.")
        target = frame[target_column].dropna()
        class_count = target.nunique()
        if 2 <= class_count <= 20:
            proportions = target.value_counts(normalize=True)
            minority_share = float(proportions.min())
            if minority_share < 0.2:
                findings.append(Finding(
                    "Class imbalance",
                    "High" if minority_share < 0.05 else "Medium",
                    f"{_clean_name(target_column)} is imbalanced",
                    f"The least frequent class represents {minority_share:.1%} of non-missing target values.",
                    "Use stratified splits and evaluate per-class metrics; consider resampling only on training data.",
                ))
                recommendations.append(
                    f"Use stratified evaluation for target {_clean_name(target_column)}; inspect per-class metrics."
                )

    for name in numeric_names:
        values = frame[name].dropna()
        if len(values) >= 3:
            skew = float(values.skew())
            if np.isfinite(skew) and abs(skew) > 2:
                findings.append(Finding(
                    "Skewed distribution",
                    "Low" if abs(skew) < 3 else "Medium",
                    f"{_clean_name(name)} is strongly skewed",
                    f"Sample skewness is {skew:.2f}; extreme values may affect some models.",
                    "Inspect the distribution; a log or other power transform may help when appropriate.",
                ))
                recommendations.append(f"Inspect the distribution of {_clean_name(name)} (skew={skew:.2f}).")

    if not findings:
        findings.append(Finding(
            "Dataset status",
            "Healthy",
            "No major issues detected",
            "The automated checks did not find common data-quality problems.",
            "Confirm feature meaning, units, and target definition before modeling.",
        ))

    missing_penalty = min(25, missing_percent * 0.7)
    duplicate_penalty = min(15, duplicate_rows / rows * 100 * 1.5)
    outlier_rate = sum(outlier_counts.values()) / max(1, rows * max(1, len(numeric_names))) * 100
    outlier_penalty = min(15, outlier_rate * 0.75)
    structural_penalty = min(15, len(constant_names) * 5)
    correlation_penalty = min(12, len(correlation_pairs) * 3)
    invalid_penalty = min(10, invalid_count * 2)
    health_score = int(round(max(
        0,
        100 - missing_penalty - duplicate_penalty - outlier_penalty
        - structural_penalty - correlation_penalty - invalid_penalty,
    )))

    class_imbalance_penalty = 0
    if target_column is not None and target_column in frame.columns:
        target = frame[target_column].dropna()
        if 2 <= target.nunique() <= 20:
            minority_share = float(target.value_counts(normalize=True).min())
            class_imbalance_penalty = 10 if minority_share < 0.05 else 5 if minority_share < 0.2 else 0
    categorical_penalty = min(10, len(categorical_names) * 2)
    ml_readiness_score = max(0, health_score - categorical_penalty - class_imbalance_penalty)

    column_summary = pd.DataFrame({
        "Column": [_clean_name(name) for name in frame.columns],
        "Type": [str(frame[name].dtype) for name in frame.columns],
        "Missing": [int(frame[name].isna().sum()) for name in frame.columns],
        "Missing %": [round(float(frame[name].isna().mean() * 100), 1) for name in frame.columns],
        "Unique": [int(frame[name].nunique(dropna=True)) for name in frame.columns],
    })

    return DatasetReport(
        rows=rows,
        columns=columns,
        numeric_columns=len(numeric_names),
        categorical_columns=len(categorical_names),
        datetime_columns=len(datetime_names),
        duplicate_rows=duplicate_rows,
        missing_cells=missing_cells,
        missing_percent=missing_percent,
        health_score=health_score,
        ml_readiness_score=ml_readiness_score,
        findings=tuple(findings),
        recommendations=tuple(dict.fromkeys(recommendations)),
        column_summary=column_summary,
    )


def prepare_cleaned_copy(
    frame: pd.DataFrame,
    *,
    impute_missing: bool = True,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """Return a copy with optional median/mode imputation and exact deduplication."""
    cleaned = frame.copy()
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    if impute_missing:
        for name in cleaned.columns:
            if not cleaned[name].isna().any():
                continue
            if is_numeric_dtype(cleaned[name].dtype) and not is_bool_dtype(cleaned[name].dtype):
                median = cleaned[name].median()
                if pd.notna(median):
                    cleaned[name] = cleaned[name].fillna(median)
            else:
                modes = cleaned[name].mode(dropna=True)
                if not modes.empty:
                    cleaned[name] = cleaned[name].fillna(modes.iloc[0])
    return cleaned

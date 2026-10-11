"""Desktop interface for the AI Dataset Health Inspector."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dataset_inspector import DatasetReport, Finding, analyze_dataframe, prepare_cleaned_copy


APP_STYLE = """
QWidget {
    background: #10151d;
    color: #e6edf5;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QMainWindow { background: #10151d; }
QLabel#muted { color: #99a8ba; }
QLabel#title { font-size: 23pt; font-weight: 700; color: #f2f6fb; }
QLabel#sectionTitle { font-size: 13pt; font-weight: 650; color: #edf3fa; }
QLabel#score { font-size: 29pt; font-weight: 700; }
QPushButton {
    background: #202a37; border: 1px solid #354355; border-radius: 8px;
    padding: 9px 15px; font-weight: 600;
}
QPushButton:hover { background: #29384a; border-color: #4d647d; }
QPushButton#primary { background: #087e8b; border: 0; color: white; }
QPushButton#primary:hover { background: #0998a6; }
QComboBox {
    background: #18212c; border: 1px solid #354355; border-radius: 6px;
    padding: 7px 10px; min-width: 170px;
}
QFrame#card { background: #171f2a; border: 1px solid #263342; border-radius: 12px; }
QFrame#finding { background: #171f2a; border: 1px solid #263342; border-radius: 9px; }
QTabWidget::pane { border: 1px solid #263342; border-radius: 8px; top: -1px; }
QTabBar::tab {
    background: #141b24; color: #9baabd; padding: 10px 18px;
    border: 0; border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: #58d3d2; border-bottom: 2px solid #16a6a6; }
QTableWidget {
    background: #151d27; alternate-background-color: #192330;
    gridline-color: #293644; border: 1px solid #263342; border-radius: 7px;
}
QHeaderView::section {
    background: #1d2936; color: #b8c8d8; border: 0; padding: 8px; font-weight: 600;
}
QScrollArea { border: 0; }
QDialog { background: #141b24; }
"""

SEVERITY_COLORS = {
    "Critical": "#ff6b6b",
    "High": "#ff9a62",
    "Medium": "#f3c969",
    "Low": "#67c7d1",
    "Healthy": "#62cf9b",
}


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    return label


def _section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


class MetricCard(QFrame):
    def __init__(self, title: str, value: str, subtitle: str = "") -> None:
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(17, 14, 17, 14)
        layout.setSpacing(5)
        layout.addWidget(_muted(title.upper()))
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 21pt; font-weight: 700;")
        layout.addWidget(self.value_label)
        self.subtitle_label = _muted(subtitle)
        layout.addWidget(self.subtitle_label)

    def set_value(self, value: str, subtitle: str | None = None) -> None:
        self.value_label.setText(value)
        if subtitle is not None:
            self.subtitle_label.setText(subtitle)


class FindingCard(QFrame):
    def __init__(self, finding: Finding) -> None:
        super().__init__()
        self.setObjectName("finding")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(6)

        top = QHBoxLayout()
        severity = QLabel(finding.severity.upper())
        severity.setStyleSheet(
            f"color: {SEVERITY_COLORS[finding.severity]}; font-weight: 700; font-size: 9pt;"
        )
        category = _muted(finding.category)
        top.addWidget(severity)
        top.addWidget(category)
        top.addStretch(1)
        layout.addLayout(top)

        title = QLabel(finding.title)
        title.setStyleSheet("font-size: 12pt; font-weight: 650;")
        title.setWordWrap(True)
        layout.addWidget(title)
        detail = QLabel(finding.detail)
        detail.setObjectName("muted")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        recommendation = QLabel(f"Recommended: {finding.recommendation}")
        recommendation.setWordWrap(True)
        recommendation.setStyleSheet("color: #83d6cd;")
        layout.addWidget(recommendation)


class ExportOptionsDialog(QDialog):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cleaned export options")
        self.setMinimumWidth(390)
        layout = QVBoxLayout(self)
        layout.addWidget(_section_title("Prepare a cleaned copy"))
        layout.addWidget(_muted("Your original dataset will not be changed."))
        self.impute_box = QCheckBox("Fill missing values (numeric median, otherwise most frequent)")
        self.impute_box.setChecked(True)
        self.duplicates_box = QCheckBox("Remove exact duplicate rows")
        self.duplicates_box.setChecked(True)
        layout.addWidget(self.impute_box)
        layout.addWidget(self.duplicates_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.export_button = buttons.addButton(
            "Choose export location", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.export_button.setObjectName("primary")
        self.export_button.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class DatasetInspectorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Dataset Health Inspector")
        self.resize(1180, 820)
        self.frame: pd.DataFrame | None = None
        self.report: DatasetReport | None = None
        self.file_path: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 22, 28, 22)
        root.setSpacing(17)

        heading = QHBoxLayout()
        title_stack = QVBoxLayout()
        title = QLabel("DATASET HEALTH INSPECTOR")
        title.setObjectName("title")
        title_stack.addWidget(title)
        self.file_label = _muted("Upload a CSV or Excel file to begin your dataset diagnosis.")
        title_stack.addWidget(self.file_label)
        heading.addLayout(title_stack)
        heading.addStretch(1)
        self.open_button = QPushButton("Upload dataset")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self.open_dataset)
        heading.addWidget(self.open_button)
        self.export_button = QPushButton("Export cleaned copy")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_cleaned)
        heading.addWidget(self.export_button)
        root.addLayout(heading)

        score_row = QHBoxLayout()
        self.health_card = MetricCard("Dataset health", "—", "Upload a dataset to calculate")
        self.readiness_card = MetricCard("ML readiness", "—", "Based on detected data risks")
        self.row_card = MetricCard("Rows", "—")
        self.column_card = MetricCard("Columns", "—")
        self.missing_card = MetricCard("Missing values", "—")
        self.duplicate_card = MetricCard("Duplicate rows", "—")
        for card in (
            self.health_card, self.readiness_card, self.row_card,
            self.column_card, self.missing_card, self.duplicate_card,
        ):
            score_row.addWidget(card, 1)
        root.addLayout(score_row)

        tools_row = QHBoxLayout()
        tools_row.addWidget(_section_title("Inspection"))
        tools_row.addStretch(1)
        tools_row.addWidget(_muted("Classification target (optional)"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        self.target_combo.currentIndexChanged.connect(self.refresh_report)
        tools_row.addWidget(self.target_combo)
        root.addLayout(tools_row)

        self.tabs = QTabWidget()
        self.findings_area = QScrollArea()
        self.findings_area.setWidgetResizable(True)
        self.findings_content = QWidget()
        self.findings_layout = QVBoxLayout(self.findings_content)
        self.findings_layout.setContentsMargins(12, 12, 12, 12)
        self.findings_layout.setSpacing(10)
        self.findings_area.setWidget(self.findings_content)
        self.tabs.addTab(self.findings_area, "Health findings")

        self.columns_table = QTableWidget(0, 5)
        self.columns_table.setHorizontalHeaderLabels(
            ["Column", "Data type", "Missing", "Missing %", "Unique"]
        )
        self._configure_table(self.columns_table)
        self.tabs.addTab(self.columns_table, "Column profile")

        self.preview_table = QTableWidget(0, 0)
        self._configure_table(self.preview_table)
        self.tabs.addTab(self.preview_table, "Data preview")
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
        self._show_empty_state()

    @staticmethod
    def _configure_table(table: QTableWidget) -> None:
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSortingEnabled(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)

    def _show_empty_state(self) -> None:
        label = QLabel("Your dataset diagnosis will appear here.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setObjectName("muted")
        self.findings_layout.addWidget(label)
        self.findings_layout.addStretch(1)

    def open_dataset(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open dataset",
            "",
            "Datasets (*.csv *.xlsx *.xls);;CSV files (*.csv);;Excel files (*.xlsx *.xls)",
        )
        if not file_name:
            return
        path = Path(file_name)
        try:
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                frame = pd.read_excel(path)
            else:
                raise ValueError("Choose a CSV or Excel file.")
            if frame.empty or frame.shape[1] == 0:
                raise ValueError("The selected file contains no tabular data.")
        except (OSError, ValueError, ImportError, pd.errors.ParserError) as error:
            QMessageBox.critical(self, "Could not open dataset", str(error))
            return

        self.frame = frame
        self.file_path = path
        self.file_label.setText(f"{path.name}  ·  {path.stat().st_size / 1024:,.1f} KB")
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItem("No target selected", None)
        for name in frame.columns:
            self.target_combo.addItem(str(name), name)
        self.target_combo.setCurrentIndex(0)
        self.target_combo.blockSignals(False)
        self.target_combo.setEnabled(True)
        self.export_button.setEnabled(True)
        self._populate_preview()
        self.refresh_report()
        self.statusBar().showMessage(f"Loaded {frame.shape[0]:,} rows × {frame.shape[1]:,} columns")

    def refresh_report(self, *_args: object) -> None:
        if self.frame is None:
            return
        target = self.target_combo.currentData()
        try:
            self.report = analyze_dataframe(self.frame, target)
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Could not analyze dataset", str(error))
            return
        report = self.report
        self.health_card.set_value(
            f"{report.health_score}/100",
            "Overall data-quality signals",
        )
        self.health_card.value_label.setStyleSheet(
            f"font-size: 29pt; font-weight: 700; color: "
            f"{'#62cf9b' if report.health_score >= 80 else '#f3c969' if report.health_score >= 60 else '#ff806d'};"
        )
        self.readiness_card.set_value(f"{report.ml_readiness_score}/100")
        self.row_card.set_value(f"{report.rows:,}", "observations")
        self.column_card.set_value(
            str(report.columns),
            f"{report.numeric_columns} numeric · {report.categorical_columns} other",
        )
        self.missing_card.set_value(
            f"{report.missing_percent:.1f}%",
            f"{report.missing_cells:,} empty cells",
        )
        self.duplicate_card.set_value(f"{report.duplicate_rows:,}", "exact duplicate rows")
        self._populate_findings(report)
        self._populate_column_summary(report)

    def _populate_findings(self, report: DatasetReport) -> None:
        while self.findings_layout.count():
            item = self.findings_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        summary = QFrame()
        summary.setObjectName("card")
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(16, 13, 16, 13)
        summary_layout.addWidget(_section_title("AI-assisted diagnosis"))
        summary_layout.addWidget(_muted(
            "Automated statistical checks identify common risks; validate each action against your domain."
        ))
        self.findings_layout.addWidget(summary)

        counts = {
            severity: sum(f.severity == severity for f in report.findings)
            for severity in ("Healthy", "Low", "Medium", "High", "Critical")
        }
        summary_counts = QLabel(
            f"  {counts['Healthy']} healthy     ·     {counts['Low']} low     ·     "
            f"{counts['Medium']} medium     ·     {counts['High']} high     ·     {counts['Critical']} critical"
        )
        summary_counts.setObjectName("muted")
        self.findings_layout.addWidget(summary_counts)
        for finding in report.findings:
            self.findings_layout.addWidget(FindingCard(finding))
        self.findings_layout.addStretch(1)

    def _populate_column_summary(self, report: DatasetReport) -> None:
        summary = report.column_summary
        self.columns_table.setRowCount(len(summary))
        for row, values in enumerate(summary.itertuples(index=False, name=None)):
            for column, value in enumerate(values):
                self.columns_table.setItem(
                    row, column, QTableWidgetItem(f"{value:,}" if isinstance(value, int) else str(value))
                )
        self.columns_table.resizeColumnsToContents()

    def _populate_preview(self) -> None:
        if self.frame is None:
            return
        preview = self.frame.head(200)
        self.preview_table.clear()
        self.preview_table.setColumnCount(len(preview.columns))
        self.preview_table.setHorizontalHeaderLabels([str(name) for name in preview.columns])
        self.preview_table.setRowCount(len(preview))
        for row in range(len(preview)):
            for column in range(len(preview.columns)):
                value = preview.iat[row, column]
                text = "" if pd.isna(value) else str(value)
                self.preview_table.setItem(row, column, QTableWidgetItem(text))
        self.preview_table.resizeColumnsToContents()
        self.tabs.setTabText(2, f"Data preview · first {len(preview):,} rows")

    def export_cleaned(self) -> None:
        if self.frame is None:
            return
        options = ExportOptionsDialog(self)
        if options.exec() != QDialog.DialogCode.Accepted:
            return
        initial_name = (
            f"{self.file_path.stem}_cleaned{self.file_path.suffix}"
            if self.file_path else "dataset_cleaned.csv"
        )
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export cleaned dataset",
            str(self.file_path.with_name(initial_name)) if self.file_path else initial_name,
            "CSV files (*.csv);;Excel files (*.xlsx)",
        )
        if not file_name:
            return
        path = Path(file_name)
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".xlsx"}:
            path = path.with_suffix(".csv")
            suffix = ".csv"
        try:
            cleaned = prepare_cleaned_copy(
                self.frame,
                impute_missing=options.impute_box.isChecked(),
                remove_duplicates=options.duplicates_box.isChecked(),
            )
            if suffix == ".csv":
                cleaned.to_csv(path, index=False)
            else:
                cleaned.to_excel(path, index=False)
        except (OSError, ValueError, ImportError) as error:
            QMessageBox.critical(self, "Could not export dataset", str(error))
            return
        remaining_missing = int(cleaned.isna().sum().sum())
        QMessageBox.information(
            self,
            "Export complete",
            f"Cleaned copy saved to:\n{path}\n\nRemaining missing cells: {remaining_missing:,}",
        )


def main() -> int:
    application = QApplication(sys.argv)
    application.setStyleSheet(APP_STYLE)
    window = DatasetInspectorWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())

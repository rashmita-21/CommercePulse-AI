
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

print("CommercePulse AI - Full Workbook Inspection")
print("=" * 50)

with pd.ExcelFile(DATA_PATH) as workbook:
    print("\nWorksheets:", workbook.sheet_names)

    for sheet in workbook.sheet_names:
        print(f"\n{'=' * 50}")
        print(f"Worksheet: {sheet}")

        df = pd.read_excel(workbook, sheet_name=sheet)

        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print("Column names:", df.columns.tolist())

        print("\nDate range:")
        print(df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())

        print("\nMissing values:")
        print(df.isna().sum())

        print("\nNegative quantities:", (df["Quantity"] < 0).sum())
        print("Zero quantities:", (df["Quantity"] == 0).sum())
        print("Zero prices:", (df["Price"] == 0).sum())

print("\nWorkbook inspection complete.")

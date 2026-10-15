
from pathlib import Path

import pandas as pd

# Locate the project and source workbook
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load both worksheets and preserve their original source
sheets = []

with pd.ExcelFile(DATA_PATH) as workbook:
    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet_name)
        df["SourceSheet"] = sheet_name
        sheets.append(df)

# Combine the two worksheets
data = pd.concat(sheets, ignore_index=True)

print("COMMERCEPULSE AI — DATA AUDIT")
print("=" * 55)

print(f"\nTotal rows: {len(data):,}")
print(f"Unique invoices: {data['Invoice'].nunique():,}")
print(f"Unique products: {data['StockCode'].nunique():,}")
print(f"Unique identified customers: {data['Customer ID'].nunique():,}")
print(f"Countries: {data['Country'].nunique():,}")

# 1. Investigate cancellation-style invoice numbers
invoice_text = data["Invoice"].astype(str).str.strip().str.upper()
cancellation_mask = invoice_text.str.startswith("C")

print("\n1. CANCELLATION-STYLE INVOICES")
print(f"Rows with invoice numbers starting with C: {cancellation_mask.sum():,}")
print(
    "Unique cancellation-style invoices:",
    data.loc[cancellation_mask, "Invoice"].nunique()
)
print(f"Negative-quantity rows: {(data['Quantity'] < 0).sum():,}")

print("\nExamples of cancellation-style records:")
print(
    data.loc[
        cancellation_mask,
        ["Invoice", "StockCode", "Quantity", "Price", "Customer ID"]
    ].head(5).to_string(index=False)
)

# 2. Investigate unusual prices and quantities
print("\n2. UNUSUAL VALUES")
print(f"Zero-price rows: {(data['Price'] == 0).sum():,}")
print(f"Negative-price rows: {(data['Price'] < 0).sum():,}")
print(f"Zero-quantity rows: {(data['Quantity'] == 0).sum():,}")

# 3. Check duplicate records
print("\n3. DUPLICATE RECORDS")
print(f"Exact duplicate rows across combined data: {data.duplicated().sum():,}")

business_columns = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country"
]
print(
    "Exact duplicate rows excluding source-sheet label:",
    data.duplicated(subset=business_columns).sum()
)

# 4. Investigate the date overlap between worksheets
print("\n4. WORKSHEET DATE OVERLAP")

start = pd.Timestamp("2010-12-01")
end = pd.Timestamp("2010-12-09 23:59:59")

overlap_period = data[
    data["InvoiceDate"].between(start, end)
]

for sheet_name, group in overlap_period.groupby("SourceSheet"):
    print(
        f"{sheet_name}: {len(group):,} rows, "
        f"{group['Invoice'].nunique():,} unique invoices"
    )

first_sheet, second_sheet = sheets[0], sheets[1]

first_invoices = set(first_sheet["Invoice"].astype(str).str.strip())
second_invoices = set(second_sheet["Invoice"].astype(str).str.strip())

shared_invoices = first_invoices.intersection(second_invoices)

print(
    "\nInvoice numbers appearing in BOTH worksheets:",
    f"{len(shared_invoices):,}"
)

print("\nAudit complete. No records have been deleted or modified.")

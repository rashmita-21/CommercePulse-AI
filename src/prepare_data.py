
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load both worksheets
frames = []

with pd.ExcelFile(DATA_PATH) as workbook:
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet)
        frame["SourceSheet"] = sheet
        frames.append(frame)

data = pd.concat(frames, ignore_index=True)

original_rows = len(data)

# 2. Remove exact duplicate transaction records.
# Keep the source sheet label for traceability.
business_columns = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country"
]

duplicate_mask = data.duplicated(subset=business_columns)
duplicate_rows = int(duplicate_mask.sum())
data = data.drop_duplicates(subset=business_columns, keep="first").copy()

# 3. Standardize text fields and create analysis columns
for column in ["Invoice", "StockCode", "Description", "Country"]:
    data[column] = data[column].astype("string").str.strip()

data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
data["IsCancellationInvoice"] = (
    data["Invoice"].str.upper().str.startswith("C", na=False)
)
data["LineValue"] = data["Quantity"] * data["Price"]
data["Month"] = data["InvoiceDate"].dt.to_period("M").astype("string")

# 4. Separate sales, returns, and cancellation-style records
sales = data[
    (~data["IsCancellationInvoice"])
    & (data["Quantity"] > 0)
    & (data["Price"] > 0)
    & data["InvoiceDate"].notna()
].copy()

returns = data[
    (data["Quantity"] < 0)
    | data["IsCancellationInvoice"]
].copy()

# Customer analysis requires an identified customer
identified_sales = sales[sales["Customer ID"].notna()].copy()

# 5. Create summary tables for Power BI
monthly_sales = (
    sales.groupby("Month", as_index=False)
    .agg(
        SalesValue=("LineValue", "sum"),
        UnitsSold=("Quantity", "sum"),
        InvoiceCount=("Invoice", "nunique"),
    )
    .sort_values("Month")
)

product_sales = (
    sales.groupby(["StockCode", "Description"], dropna=False, as_index=False)
    .agg(
        SalesValue=("LineValue", "sum"),
        UnitsSold=("Quantity", "sum"),
        InvoiceCount=("Invoice", "nunique"),
    )
    .sort_values("SalesValue", ascending=False)
)

country_sales = (
    sales.groupby("Country", as_index=False)
    .agg(
        SalesValue=("LineValue", "sum"),
        UnitsSold=("Quantity", "sum"),
        InvoiceCount=("Invoice", "nunique"),
    )
    .sort_values("SalesValue", ascending=False)
)

customer_summary = (
    identified_sales.groupby("Customer ID", as_index=False)
    .agg(
        SalesValue=("LineValue", "sum"),
        Orders=("Invoice", "nunique"),
        UnitsPurchased=("Quantity", "sum"),
        LastPurchase=("InvoiceDate", "max"),
    )
)

customer_summary["AverageOrderValue"] = (
    customer_summary["SalesValue"] / customer_summary["Orders"]
)

# 6. Save outputs
data.to_csv(OUTPUT_DIR / "transactions_deduplicated.csv", index=False)
sales.to_csv(OUTPUT_DIR / "sales_lines.csv", index=False)
returns.to_csv(OUTPUT_DIR / "returns_and_cancellations.csv", index=False)

monthly_sales.to_csv(REPORT_DIR / "monthly_sales.csv", index=False)
product_sales.to_csv(REPORT_DIR / "product_sales.csv", index=False)
country_sales.to_csv(REPORT_DIR / "country_sales.csv", index=False)
customer_summary.to_csv(REPORT_DIR / "customer_summary.csv", index=False)

# 7. Save a simple processing report
summary = pd.DataFrame({
    "Metric": [
        "Original rows",
        "Exact duplicate rows removed",
        "Rows after deduplication",
        "Sales lines retained",
        "Return/cancellation-related rows",
        "Sales rows with identified customers",
        "Unique products in sales",
        "Unique invoices in sales",
        "Identified customers in sales",
    ],
    "Value": [
        original_rows,
        duplicate_rows,
        len(data),
        len(sales),
        len(returns),
        len(identified_sales),
        sales["StockCode"].nunique(),
        sales["Invoice"].nunique(),
        identified_sales["Customer ID"].nunique(),
    ],
})

summary.to_csv(REPORT_DIR / "processing_summary.csv", index=False)

print("COMMERCEPULSE AI — DATA PREPARATION COMPLETE")
print("=" * 50)
print(summary.to_string(index=False))
print("\nCreated files:")
print("- data/processed/transactions_deduplicated.csv")
print("- data/processed/sales_lines.csv")
print("- data/processed/returns_and_cancellations.csv")
print("- reports/monthly_sales.csv")
print("- reports/product_sales.csv")
print("- reports/country_sales.csv")
print("- reports/customer_summary.csv")
print("- reports/processing_summary.csv")
print("\nNote: SalesValue excludes cancellation-style invoices,")
print("negative quantities, and zero/negative prices.")

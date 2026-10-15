
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sales_lines.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sales = pd.read_csv(
    DATA_PATH,
    parse_dates=["InvoiceDate"],
    dtype={"Invoice": "string"},
    low_memory=False,
)

# Create month and line-level sales value
sales["Month"] = sales["InvoiceDate"].dt.to_period("M").astype(str)
sales["LineValue"] = sales["Quantity"] * sales["Price"]

# 1. Overall business KPIs
identified = sales[sales["Customer ID"].notna()].copy()

kpis = pd.DataFrame({
    "Metric": [
        "Sales value (before returns are deducted)",
        "Units sold",
        "Unique sales invoices",
        "Unique products sold",
        "Countries served",
        "Identified customers",
        "Average invoice value (identified customers only)",
    ],
    "Value": [
        sales["LineValue"].sum(),
        sales["Quantity"].sum(),
        sales["Invoice"].nunique(),
        sales["StockCode"].nunique(),
        sales["Country"].nunique(),
        identified["Customer ID"].nunique(),
        (
            identified.groupby("Invoice")["LineValue"].sum().mean()
            if not identified.empty else 0
        ),
    ],
})

# 2. Monthly sales trends
monthly = (
    sales.groupby("Month", as_index=False)
    .agg(
        SalesValue=("LineValue", "sum"),
        UnitsSold=("Quantity", "sum"),
        Orders=("Invoice", "nunique"),
    )
    .sort_values("Month")
)

monthly["PreviousMonthSales"] = monthly["SalesValue"].shift(1)
monthly["SalesGrowthPct"] = (
    (monthly["SalesValue"] - monthly["PreviousMonthSales"])
    / monthly["PreviousMonthSales"].abs().replace(0, pd.NA)
) * 100

# 3. Top products
products = (
    sales.groupby(["StockCode", "Description"], dropna=False, as_index=False)
    .agg(
        SalesValue=("LineValue", "sum"),
        UnitsSold=("Quantity", "sum"),
        Orders=("Invoice", "nunique"),
    )
    .sort_values("SalesValue", ascending=False)
)

# 4. Country performance
countries = (
    sales.groupby("Country", as_index=False)
    .agg(
        SalesValue=("LineValue", "sum"),
        UnitsSold=("Quantity", "sum"),
        Orders=("Invoice", "nunique"),
    )
    .sort_values("SalesValue", ascending=False)
)

# 5. Customer repeat-purchase behaviour
customer_orders = (
    identified.groupby("Customer ID")["Invoice"]
    .nunique()
)

customer_summary = (
    identified.groupby("Customer ID", as_index=False)
    .agg(
        SalesValue=("LineValue", "sum"),
        Orders=("Invoice", "nunique"),
        LastPurchase=("InvoiceDate", "max"),
    )
)

customer_summary["RepeatCustomer"] = customer_summary["Orders"] >= 2

repeat_rate = (
    customer_summary["RepeatCustomer"].mean() * 100
    if not customer_summary.empty else 0
)

# 6. Export the results
kpis.to_csv(REPORT_DIR / "business_kpis.csv", index=False)
monthly.to_csv(REPORT_DIR / "monthly_trends.csv", index=False)
products.to_csv(REPORT_DIR / "top_products.csv", index=False)
countries.to_csv(REPORT_DIR / "country_performance.csv", index=False)
customer_summary.to_csv(REPORT_DIR / "customer_behaviour.csv", index=False)

print("COMMERCEPULSE AI — BUSINESS INSIGHTS")
print("=" * 50)

print("\nOVERALL KPIs")
print(kpis.to_string(index=False))

print("\nTOP 10 PRODUCTS BY SALES VALUE")
print(products.head(10).to_string(index=False))

print("\nTOP 10 COUNTRIES BY SALES VALUE")
print(countries.head(10).to_string(index=False))

print("\nTOP 5 MONTHS BY SALES VALUE")
print(
    monthly.nlargest(5, "SalesValue")[
        ["Month", "SalesValue", "Orders"]
    ].to_string(index=False)
)

print(f"\nRepeat-customer rate: {repeat_rate:.2f}%")
print("Definition: identified customers with at least 2 distinct invoices.")

print("\nInsight files saved in the reports folder.")

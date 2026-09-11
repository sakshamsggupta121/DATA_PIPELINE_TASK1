import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# E-COMMERCE DATA PIPELINE
# Elite Tech Intern - Data Science Internship
# Task 1: Data Pipeline Development
# ============================================================


# -----------------------------
# 1. PROJECT PATHS
# -----------------------------

DATA_DIR = "data"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# 2. LOAD DATA
# -----------------------------

print("\n" + "=" * 60)
print("1. LOADING RAW DATA")
print("=" * 60)

orders = pd.read_csv(
    os.path.join(DATA_DIR, "List of Orders.csv")
)

order_details = pd.read_csv(
    os.path.join(DATA_DIR, "Order Details.csv")
)

sales_target = pd.read_csv(
    os.path.join(DATA_DIR, "Sales target.csv")
)

print("List of Orders:", orders.shape)
print("Order Details:", order_details.shape)
print("Sales Target:", sales_target.shape)


# -----------------------------
# 3. CLEAN COLUMN NAMES
# -----------------------------

def clean_column_names(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


orders = clean_column_names(orders)
order_details = clean_column_names(order_details)
sales_target = clean_column_names(sales_target)


print("\nColumns after standardization:")
print("Orders:", orders.columns.tolist())
print("Order Details:", order_details.columns.tolist())
print("Sales Target:", sales_target.columns.tolist())


# -----------------------------
# 4. REMOVE DUPLICATES
# -----------------------------

print("\n" + "=" * 60)
print("2. REMOVING DUPLICATES")
print("=" * 60)

print("Orders duplicates:", orders.duplicated().sum())
print("Order Details duplicates:", order_details.duplicated().sum())
print("Sales Target duplicates:", sales_target.duplicated().sum())

orders = orders.drop_duplicates()
order_details = order_details.drop_duplicates()
sales_target = sales_target.drop_duplicates()


# -----------------------------
# 5. HANDLE MISSING VALUES
# -----------------------------

print("\n" + "=" * 60)
print("3. HANDLING MISSING VALUES")
print("=" * 60)


def handle_missing_values(df):

    # Numeric columns
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        if df[column].isnull().any():
            df[column] = df[column].fillna(
                df[column].median()
            )

    # Text columns
    text_columns = df.select_dtypes(
        include="object"
    ).columns

    for column in text_columns:

        if df[column].isnull().any():
            df[column] = df[column].fillna(
                "Unknown"
            )

    return df


orders = handle_missing_values(orders)
order_details = handle_missing_values(order_details)
sales_target = handle_missing_values(sales_target)


print("Missing values handled successfully.")


# -----------------------------
# 6. DATE CONVERSION
# -----------------------------

print("\n" + "=" * 60)
print("4. CONVERTING DATA TYPES")
print("=" * 60)


for column in orders.columns:

    if "date" in column:

        orders[column] = pd.to_datetime(
            orders[column],
            errors="coerce"
        )

        print(f"Converted {column} to datetime.")


# -----------------------------
# 7. NUMERIC CONVERSION
# -----------------------------

numeric_columns = [
    "amount",
    "quantity",
    "profit"
]

for column in numeric_columns:

    if column in order_details.columns:

        order_details[column] = pd.to_numeric(
            order_details[column],
            errors="coerce"
        )


# -----------------------------
# 8. REMOVE INVALID VALUES
# -----------------------------

print("\n" + "=" * 60)
print("5. REMOVING INVALID VALUES")
print("=" * 60)


if "amount" in order_details.columns:

    order_details = order_details[
        order_details["amount"] >= 0
    ]


if "quantity" in order_details.columns:

    order_details = order_details[
        order_details["quantity"] > 0
    ]


print("Invalid records removed.")


# -----------------------------
# 9. STANDARDIZE ORDER ID
# -----------------------------

print("\n" + "=" * 60)
print("6. MERGING DATASETS")
print("=" * 60)


# Find Order ID column automatically

order_id_orders = [
    col for col in orders.columns
    if "order" in col and "id" in col
]

order_id_details = [
    col for col in order_details.columns
    if "order" in col and "id" in col
]


if not order_id_orders:

    raise ValueError(
        "Order ID column not found in List of Orders.csv"
    )


if not order_id_details:

    raise ValueError(
        "Order ID column not found in Order Details.csv"
    )


orders = orders.rename(
    columns={
        order_id_orders[0]: "order_id"
    }
)


order_details = order_details.rename(
    columns={
        order_id_details[0]: "order_id"
    }
)


# -----------------------------
# 10. MERGE ORDERS + DETAILS
# -----------------------------

merged_data = pd.merge(
    order_details,
    orders,
    on="order_id",
    how="left"
)


print(
    "Merged dataset shape:",
    merged_data.shape
)


# -----------------------------
# 11. FEATURE ENGINEERING
# -----------------------------

print("\n" + "=" * 60)
print("7. FEATURE ENGINEERING")
print("=" * 60)


# Find date column

date_column = None

for column in merged_data.columns:

    if "date" in column:

        date_column = column
        break


if date_column:

    merged_data["year"] = (
        merged_data[date_column].dt.year
    )

    merged_data["month"] = (
        merged_data[date_column].dt.month
    )

    merged_data["month_name"] = (
        merged_data[date_column].dt.month_name()
    )


# Profit margin

if (
    "amount" in merged_data.columns
    and "profit" in merged_data.columns
):

    merged_data["profit_margin"] = np.where(
        merged_data["amount"] != 0,
        (
            merged_data["profit"]
            / merged_data["amount"]
        ) * 100,
        0
    )


print("New features created.")


# -----------------------------
# 12. DATA VALIDATION
# -----------------------------

print("\n" + "=" * 60)
print("8. DATA VALIDATION")
print("=" * 60)


print("Rows:", len(merged_data))
print(
    "Duplicate rows:",
    merged_data.duplicated().sum()
)


print(
    "Total missing values:",
    merged_data.isnull().sum().sum()
)


# -----------------------------
# 13. CREATE CATEGORY SUMMARY
# -----------------------------

print("\n" + "=" * 60)
print("9. CREATING SALES SUMMARY")
print("=" * 60)


if "category" in merged_data.columns:

    summary = (
        merged_data
        .groupby("category")
        .agg(
            total_sales=("amount", "sum"),
            total_profit=("profit", "sum"),
            total_quantity=("quantity", "sum"),
            number_of_orders=("order_id", "nunique")
        )
        .reset_index()
    )

else:

    summary = pd.DataFrame()


# -----------------------------
# 14. SCALING NUMERIC FEATURES
# -----------------------------

print("\n" + "=" * 60)
print("10. SCALING NUMERIC FEATURES")
print("=" * 60)


scaler_columns = []

for column in [
    "amount",
    "quantity",
    "profit"
]:

    if column in merged_data.columns:

        scaler_columns.append(column)


if scaler_columns:

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(
        merged_data[scaler_columns]
    )

    for i, column in enumerate(scaler_columns):

        merged_data[
            column + "_scaled"
        ] = scaled_values[:, i]


    print(
        "Scaled columns:",
        scaler_columns
    )


# -----------------------------
# 15. SAVE CLEAN DATA
# -----------------------------

print("\n" + "=" * 60)
print("11. SAVING OUTPUT DATA")
print("=" * 60)


clean_data_path = os.path.join(
    OUTPUT_DIR,
    "clean_ecommerce_data.csv"
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "sales_summary.csv"
)


merged_data.to_csv(
    clean_data_path,
    index=False
)


if not summary.empty:

    summary.to_csv(
        summary_path,
        index=False
    )


print(
    "Clean dataset saved:",
    clean_data_path
)


if not summary.empty:

    print(
        "Sales summary saved:",
        summary_path
    )


# -----------------------------
# 16. FINAL STATUS
# -----------------------------

print("\n" + "=" * 60)
print("DATA PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    """
Pipeline:
Load
  ↓
Clean
  ↓
Transform
  ↓
Merge
  ↓
Feature Engineering
  ↓
Scaling
  ↓
Validation
  ↓
Export
"""
)

print("Task 1 completed successfully!")
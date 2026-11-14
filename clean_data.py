from pathlib import Path
import pandas as pd
import numpy as np

# مجلد البيانات الأصلية
data_folder = Path("data")

# مجلد البيانات بعد التنظيف
clean_folder = Path("cleaned_data")
clean_folder.mkdir(exist_ok=True)

csv_files = sorted(data_folder.glob("*.csv"))

print("=" * 75)
print("CIC-IDS2017 DATA CLEANING")
print("=" * 75)

total_before = 0
total_after = 0

for file in csv_files:

    print("\n" + "=" * 75)
    print(f"FILE: {file.name}")
    print("=" * 75)

    # قراءة الملف
    try:
        df = pd.read_csv(
            file,
            low_memory=False,
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        print("UTF-8 failed -> using cp1252")

        df = pd.read_csv(
            file,
            low_memory=False,
            encoding="cp1252"
        )

    # تنظيف أسماء الأعمدة
    df.columns = df.columns.str.strip()

    before = len(df)
    total_before += before

    print(f"Original Rows: {before:,}")

    # ---------------------------------------
    # 1. حذف الصفوف الفارغة بالكامل
    # ---------------------------------------

    empty_rows = df.isna().all(axis=1).sum()

    df.dropna(
        how="all",
        inplace=True
    )

    print(f"Empty Rows Removed: {empty_rows:,}")

    # ---------------------------------------
    # 2. تحويل Infinity إلى NaN
    # ---------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    infinity_count = (
        np.isinf(df[numeric_columns])
        .sum()
        .sum()
    )

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    print(f"Infinity Values Found: {infinity_count:,}")

    # ---------------------------------------
    # 3. حذف الصفوف التي ما زالت تحتوي NaN
    # ---------------------------------------

    missing_before = df.isna().sum().sum()

    rows_before_nan = len(df)

    df.dropna(inplace=True)

    removed_nan_rows = (
        rows_before_nan - len(df)
    )

    print(f"Missing Values Found: {missing_before:,}")
    print(f"Rows Removed Due To Missing Values: {removed_nan_rows:,}")

    # ---------------------------------------
    # 4. إزالة الصفوف المكررة
    # ---------------------------------------

    duplicates = df.duplicated().sum()

    df.drop_duplicates(inplace=True)

    print(f"Duplicate Rows Removed: {duplicates:,}")

    # ---------------------------------------
    # 5. تنظيف Label
    # ---------------------------------------

    if "Label" not in df.columns:

        print("WARNING: Label column not found!")

        continue

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------
    # 6. إنشاء Binary Label
    #
    # BENIGN = 0
    # ATTACK = 1
    # ---------------------------------------

    df["BinaryLabel"] = np.where(
        df["Label"].str.upper() == "BENIGN",
        0,
        1
    )

    # ---------------------------------------
    # عرض النتائج
    # ---------------------------------------

    after = len(df)

    total_after += after

    print(f"\nClean Rows: {after:,}")

    print("\nLabels:")

    print(
        df["Label"]
        .value_counts()
    )

    print("\nBinary Classification:")

    print(
        df["BinaryLabel"]
        .value_counts()
    )

    # ---------------------------------------
    # حفظ الملف النظيف
    # ---------------------------------------

    output_file = (
        clean_folder /
        f"clean_{file.name}"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nSaved -> {output_file}"
    )


print("\n" + "=" * 75)

print("DATA CLEANING COMPLETED")

print("=" * 75)

print(
    f"\nTotal Rows Before Cleaning: "
    f"{total_before:,}"
)

print(
    f"Total Rows After Cleaning : "
    f"{total_after:,}"
)

print(
    f"Total Rows Removed        : "
    f"{total_before - total_after:,}"
)
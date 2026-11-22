
"""
OSMI Mental Health in Tech Survey
CRISP-DM Phase 2: Data Understanding

This script:
1. Downloads the dataset automatically from Kaggle
2. Examines dataset structure and data types
3. Profiles all columns
4. Checks missing values, duplicates and invalid ages
5. Calculates five essential survey KPIs
6. Performs exploratory data analysis (EDA)
7. Analyzes survey submission timestamps
8. Exports Excel, CSV and PNG reports

The original dataset is not modified.
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

from pathlib import Path

# Save results beside this Python script
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "OSMI_Analysis_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nOSMI MENTAL HEALTH SURVEY")
print("CRISP-DM: DATA UNDERSTANDING")
print("=" * 55)


# ============================================================
# 2. DOWNLOAD AND LOAD DATASET
# ============================================================

print("\n[1] Downloading dataset from Kaggle...")

dataset_path = Path(
    kagglehub.dataset_download(
        "osmi/mental-health-in-tech-survey"
    )
)

# Locate survey.csv automatically
csv_files = list(dataset_path.rglob("survey.csv"))

if not csv_files:
    raise FileNotFoundError(
        "survey.csv was not found in the downloaded dataset."
    )

csv_file = csv_files[0]

# Read the dataset
df = pd.read_csv(csv_file)

print("Dataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Save an unchanged copy for reference
df.to_csv(
    OUTPUT_DIR / "original_survey.csv",
    index=False
)


# ============================================================
# 3. DATA PROFILING
# ============================================================

print("\n[2] Performing data profiling...")

print("\nFirst five records:")
print(df.head().to_string())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset dimensions:")
print(df.shape)

# Generate a profile of every column
profile = pd.DataFrame({
    "Column": df.columns,
    "Data Type": [str(t) for t in df.dtypes],
    "Non Null Count": df.notna().sum().values,
    "Missing Count": df.isna().sum().values,
    "Missing Percentage": (
        df.isna().mean() * 100
    ).round(2).values,
    "Unique Values": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ]
})

profile.to_csv(
    OUTPUT_DIR / "data_profile.csv",
    index=False
)

print("\nData profiling completed.")


# ============================================================
# 4. DATA QUALITY ASSESSMENT
# ============================================================

print("\n[3] Checking data quality...")

# Check missing values
missing_values = pd.DataFrame({
    "Column": df.columns,
    "Missing Count": df.isna().sum().values,
    "Missing Percentage": (
        df.isna().mean() * 100
    ).round(2).values
})

print("\nMissing values:")
print(missing_values.to_string(index=False))

# Check exact duplicate records
duplicate_count = int(df.duplicated().sum())

print("\nExact duplicate records:", duplicate_count)

# Save duplicates for inspection
duplicates = df[df.duplicated(keep=False)]

duplicates.to_csv(
    OUTPUT_DIR / "duplicate_records.csv",
    index=False
)

# Check invalid ages
invalid_ages = pd.DataFrame()

if "Age" in df.columns:

    # Convert age to numeric for analysis only
    age_numeric = pd.to_numeric(
        df["Age"],
        errors="coerce"
    )

    # Flag potentially invalid ages
    invalid_mask = (
        age_numeric.isna()
        | (age_numeric < 16)
        | (age_numeric > 100)
    )

    invalid_ages = df.loc[invalid_mask].copy()

    print("\nPotentially invalid ages:")
    print(invalid_ages["Age"].value_counts(dropna=False))

    invalid_ages.to_csv(
        OUTPUT_DIR / "invalid_ages.csv",
        index=False
    )

# Inspect inconsistent gender values
gender_values = pd.DataFrame()

if "Gender" in df.columns:

    gender_values = (
        df["Gender"]
        .value_counts(dropna=False)
        .rename_axis("Gender")
        .reset_index(name="Count")
    )

    print("\nGender categories:")
    print(gender_values.to_string(index=False))

    gender_values.to_csv(
        OUTPUT_DIR / "gender_categories.csv",
        index=False
    )

# Summarize data quality findings
quality_summary = pd.DataFrame({
    "Metric": [
        "Total records",
        "Total columns",
        "Exact duplicate records",
        "Potentially invalid age records",
        "Columns containing missing values"
    ],
    "Value": [
        len(df),
        len(df.columns),
        duplicate_count,
        len(invalid_ages),
        int((df.isna().sum() > 0).sum())
    ]
})

print("\nData quality assessment completed.")


# ============================================================
# 5. KPI ANALYSIS
# ============================================================

print("\n[4] Calculating mental health survey KPIs...")

# All KPI calculations exclude missing responses.
# They are descriptive survey indicators, not diagnoses.

def calculate_kpi(column, positive_values, name):
    """
    Calculate percentage of valid responses matching
    the specified positive categories.
    """

    if column not in df.columns:
        return {
            "KPI": name,
            "Percentage": np.nan,
            "Positive Responses": 0,
            "Valid Responses": 0,
            "Missing Responses": len(df)
        }

    responses = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    positives = [v.lower() for v in positive_values]

    # Exclude missing and unrecognized responses
    valid_categories = {
        "yes", "no", "maybe", "don't know",
        "not sure", "never", "rarely",
        "sometimes", "often"
    }

    valid = responses.isin(valid_categories)
    positive = valid & responses.isin(positives)

    denominator = int(valid.sum())
    numerator = int(positive.sum())

    percentage = (
        round(numerator / denominator * 100, 2)
        if denominator > 0
        else np.nan
    )

    return {
        "KPI": name,
        "Percentage": percentage,
        "Positive Responses": numerator,
        "Valid Responses": denominator,
        "Missing Responses": int((~valid).sum())
    }


kpi_results = [
    calculate_kpi(
        "treatment",
        ["yes"],
        "Treatment Rate"
    ),

    calculate_kpi(
        "work_interfere",
        ["sometimes", "often"],
        "Work Interference Rate"
    ),

    calculate_kpi(
        "family_history",
        ["yes"],
        "Family History Rate"
    ),

    calculate_kpi(
        "benefits",
        ["yes"],
        "Employer Benefits Coverage"
    ),

    calculate_kpi(
        "mental_health_consequence",
        ["yes"],
        "Mental Health Consequence Concern Rate"
    )
]

kpi_df = pd.DataFrame(kpi_results)

print("\nEssential KPIs:")
print(kpi_df.to_string(index=False))

kpi_df.to_csv(
    OUTPUT_DIR / "mental_health_kpis.csv",
    index=False
)


# ============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n[5] Performing exploratory data analysis...")


def save_bar_chart(series, title, filename):
    """
    Create a bar chart showing categorical counts.
    """

    counts = series.value_counts(dropna=False)

    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Number of Respondents")

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150
    )

    plt.close(fig)


# Chart 1: Treatment distribution
if "treatment" in df.columns:

    save_bar_chart(
        df["treatment"],
        "Mental Health Treatment Responses",
        "treatment_distribution.png"
    )


# Chart 2: Work interference
if "work_interfere" in df.columns:

    save_bar_chart(
        df["work_interfere"],
        "Mental Health Interference with Work",
        "work_interference.png"
    )


# Chart 3: Employer benefits
if "benefits" in df.columns:

    save_bar_chart(
        df["benefits"],
        "Employer Mental Health Benefits",
        "employer_benefits.png"
    )


# Chart 4: Age distribution
if "Age" in df.columns:

    age = pd.to_numeric(
        df["Age"],
        errors="coerce"
    )

    # Exclude implausible ages from visualization
    age = age.where(age.between(16, 100)).dropna()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(age, bins=20)

    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Respondents")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "age_distribution.png",
        dpi=150
    )

    plt.close(fig)


# Chart 5: Family history
if "family_history" in df.columns:

    save_bar_chart(
        df["family_history"],
        "Family History of Mental Health Conditions",
        "family_history.png"
    )


# ============================================================
# 7. ANALYZE TREATMENT BY SELECTED FACTORS
# ============================================================

print("\n[6] Analyzing treatment by workplace factors...")

factor_results = []

if "treatment" in df.columns:

    # Normalize treatment responses for comparison
    treatment = (
        df["treatment"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    for factor in [
        "family_history",
        "benefits",
        "remote_work",
        "no_employees",
        "work_interfere"
    ]:

        if factor not in df.columns:
            continue

        # Only use valid Yes/No treatment responses
        temp = pd.DataFrame({
            "Group": df[factor],
            "Treatment": treatment
        })

        temp = temp[
            temp["Treatment"].isin(["yes", "no"])
            & temp["Group"].notna()
        ].copy()

        temp["Treatment Yes"] = (
            temp["Treatment"] == "yes"
        ).astype(int)

        grouped = (
            temp.groupby("Group", dropna=False)
            .agg(
                Respondents=("Treatment", "size"),
                Treatment_Yes=("Treatment Yes", "sum")
            )
            .reset_index()
        )

        grouped["Treatment Rate (%)"] = (
            grouped["Treatment_Yes"]
            / grouped["Respondents"]
            * 100
        ).round(2)

        grouped.insert(0, "Factor", factor)

        factor_results.append(grouped)

if factor_results:

    treatment_by_factor = pd.concat(
        factor_results,
        ignore_index=True
    )

else:

    treatment_by_factor = pd.DataFrame(
        columns=[
            "Factor", "Group", "Respondents",
            "Treatment_Yes", "Treatment Rate (%)"
        ]
    )

treatment_by_factor.to_csv(
    OUTPUT_DIR / "treatment_by_factor.csv",
    index=False
)


# ============================================================
# 8. TIMESTAMP ANALYSIS
# ============================================================

print("\n[7] Analyzing survey timestamps...")

monthly_summary = pd.DataFrame()

if "Timestamp" in df.columns:

    # Parse submission timestamps
    timestamps = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
        utc=True
    )

    print(
        "Invalid timestamps:",
        int(timestamps.isna().sum())
    )

    print(
        "Earliest submission:",
        timestamps.min()
    )

    print(
        "Latest submission:",
        timestamps.max()
    )

    # Aggregate survey responses by month
    time_data = pd.DataFrame({
        "Timestamp": timestamps,
        "Treatment": (
            df["treatment"]
            if "treatment" in df.columns
            else pd.Series(pd.NA, index=df.index)
        )
    })

    time_data = time_data.dropna(
        subset=["Timestamp"]
    ).copy()

    # Convert to month labels
    time_data["Month"] = (
        time_data["Timestamp"]
        .dt.strftime("%Y-%m")
    )

    # Normalize treatment responses
    time_data["Treatment"] = (
        time_data["Treatment"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    time_data["Valid Treatment"] = (
        time_data["Treatment"].isin(["yes", "no"])
    )

    time_data["Treatment Yes"] = (
        time_data["Treatment"] == "yes"
    ).astype(int)

    # Group responses by survey submission month
    monthly_summary = (
        time_data.groupby("Month")
        .agg(
            Total_Responses=("Timestamp", "size"),
            Valid_Treatment_Responses=(
                "Valid Treatment", "sum"
            ),
            Treatment_Yes=("Treatment Yes", "sum")
        )
        .reset_index()
    )

    monthly_summary["Treatment Rate (%)"] = (
        monthly_summary["Treatment_Yes"]
        / monthly_summary[
            "Valid_Treatment_Responses"
        ].replace(0, np.nan)
        * 100
    ).round(2)

    monthly_summary.to_csv(
        OUTPUT_DIR / "monthly_summary.csv",
        index=False
    )

    print("\nMonthly summary:")
    print(monthly_summary.to_string(index=False))

    # Chart: Number of responses by month
    if not monthly_summary.empty:

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            monthly_summary["Month"],
            monthly_summary["Total_Responses"],
            marker="o"
        )

        ax.set_title("Survey Responses by Month")
        ax.set_xlabel("Submission Month")
        ax.set_ylabel("Number of Responses")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "monthly_responses.png",
            dpi=150
        )

        plt.close(fig)

        # Plot monthly treatment reporting rate
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            monthly_summary["Month"],
            monthly_summary["Treatment Rate (%)"],
            marker="o"
        )

        ax.set_title(
            "Treatment Reporting Rate by Survey Month"
        )

        ax.set_xlabel("Submission Month")
        ax.set_ylabel("Treatment Reporting Rate (%)")

        ax.set_ylim(0, 100)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "monthly_treatment_rate.png",
            dpi=150
        )

        plt.close(fig)


# ============================================================
# 9. EXPORT COMPLETE EXCEL REPORT
# ============================================================

print("\n[8] Creating Excel report...")

excel_file = OUTPUT_DIR / "OSMI_Data_Understanding.xlsx"

with pd.ExcelWriter(
    excel_file,
    engine="openpyxl"
) as writer:

    profile.to_excel(
        writer,
        sheet_name="Data Profile",
        index=False
    )

    missing_values.to_excel(
        writer,
        sheet_name="Missing Values",
        index=False
    )

    quality_summary.to_excel(
        writer,
        sheet_name="Data Quality",
        index=False
    )

    kpi_df.to_excel(
        writer,
        sheet_name="KPIs",
        index=False
    )

    gender_values.to_excel(
        writer,
        sheet_name="Gender Categories",
        index=False
    )

    treatment_by_factor.to_excel(
        writer,
        sheet_name="Treatment by Factor",
        index=False
    )

    monthly_summary.to_excel(
        writer,
        sheet_name="Monthly Analysis",
        index=False
    )

print("Excel report created successfully.")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("DATA UNDERSTANDING COMPLETED")
print("=" * 55)

print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("Duplicate records:", duplicate_count)
print("Potentially invalid ages:", len(invalid_ages))

print("\nGenerated outputs:")
print("- Data profile")
print("- Missing value analysis")
print("- Data quality summary")
print("- Essential KPIs")
print("- Gender category analysis")
print("- Treatment by workplace factors")
print("- Monthly timestamp analysis")
print("- EDA charts")
print("- Excel report")

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nIMPORTANT:")
print(
    "These are survey-based indicators, not clinical "
    "mental health scores."
)
print(
    "Timestamp analysis describes submission patterns, "
    "not changes in individual employee mental health."
)
print(
    "Data cleaning and model preparation belong to "
    "the next CRISP-DM phase."
)

print("\nAnalysis finished successfully.")

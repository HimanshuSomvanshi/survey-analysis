# OSMI Mental Health in Tech Survey Project

## Project overview

This project analyzes workplace factors associated with mental-health treatment among technology employees using the OSMI Mental Health in Tech Survey.

The machine-learning objective is to estimate whether an individual has sought mental-health treatment based on survey responses.

**Target variable:** `treatment`

**Prediction:** `Yes` or `No`

**Problem type:** Binary classification

This project is for academic and organizational awareness purposes. The model is not a medical diagnosis and should not replace professional mental-health assessment.

## Current progress

| Phase | Status | Main deliverable |
|---|---|---|
| Business and data understanding | Complete | Dataset profiling and KPI analysis |
| Data preparation | Complete | Cleaned dataset |
| Exploratory data analysis | Complete | Notebooks and exported charts/results |
| Model training | Complete | Final model, test results, and metadata |
| Streamlit application | Planned | Employee-facing prediction page |
| Validation and responsible-AI review | Pending | Clinical/legal/privacy review and pilot testing |
| Deployment | Pending | Hosting and access controls |

## Project structure

```text
Documents/
  Reports, PowerPoint presentations, and supporting PDFs

data/
  survey.csv
  processed/
    mental_health_survey_cleaned.csv
    test_set.csv

notebooks/
  01_eda.ipynb
  02_preprocessing.ipynb
  03_model_training.ipynb
  OSMI_Analysis_Results/

models/
  best_model.joblib
  best_model_meta.json

Sajil's files/
  Supporting data-understanding exports and workbook
```

## Dataset

The raw OSMI dataset contains 1,259 records and 27 original fields.

Source: [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

Before redistribution or commercial use, confirm the current dataset license and terms on Kaggle. Survey comments may contain personal opinions or sensitive personal information, so use the comments field only for approved research purposes and do not collect or display employee free text in the Streamlit MVP.

The processed analytical dataset contains 1,258 records after duplicate removal, with:

- Numerical field: `age`
- 21 categorical features used in the final model
- Binary target: `treatment`

Important missing-value patterns include:

| Field | Missing percentage |
|---|---:|
| `comments` | 86.97% |
| `state` | 40.91% |
| `work_interfere` | 20.97% |
| `self_employed` | 1.43% |

Free-text comments and state are not included in the final fair-feature model.

## Data understanding and key findings

- Overall treatment rate: **50.6%**
- Family history:
  - Without family history: **35.46%** treatment rate
  - With family history: **74.19%** treatment rate
- Employer benefits:
  - Benefits available: **63.94%** treatment rate
  - Benefits unavailable: **48.40%** treatment rate
- Work interference:
  - Never: **14.08%**
  - Rarely: **70.52%**
  - Sometimes: **76.99%**
  - Often: **85.42%**

The `work_interfere` field was excluded from the final model because it is too close to a restatement of the target and could create an overly optimistic proxy result.

Key exports are located in `notebooks/OSMI_Analysis_Results/`.

## Modelling workflow

The final model is implemented in [03_model_training.ipynb](notebooks/03_model_training.ipynb).

The workflow:

1. Load and validate the processed dataset.
2. Remove exact duplicates.
3. Convert implausible ages to missing values.
4. Use a fixed 75% training and 25% testing split.
5. Maintain the positive/negative target ratio using stratification.
6. Apply median age imputation and standardization.
7. Apply categorical missing-value handling and one-hot encoding.
8. Keep preprocessing inside cross-validation pipelines to prevent leakage.
9. Compare candidate models using repeated 5-fold cross-validation.
10. Tune shortlisted models using training data only.
11. Select the final model.
12. Optimize the probability threshold using training cross-validation predictions.
13. Evaluate once on the held-out test set.
14. Save the complete fitted pipeline and metadata.

## Candidate-model comparison

The notebook compared 10 candidate approaches:

- Dummy classifier
- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradient Boosting
- SVC
- XGBoost

Repeated cross-validation results are available in [model_comparison_cv.csv](notebooks/OSMI_Analysis_Results/model_comparison_cv.csv).

Hyperparameter tuning was applied to:

- Logistic Regression
- Random Forest
- Extra Trees
- HistGradient Boosting
- XGBoost

Tuning results are available in [tuning_results.csv](notebooks/OSMI_Analysis_Results/tuning_results.csv).

## Final model

The selected final model is **Logistic Regression**.

The final model excludes `work_interfere` and uses 22 survey features.

Final held-out test performance:

| Metric | Default 0.50 threshold | Optimized 0.425 threshold |
|---|---:|---:|
| Accuracy | 70.48% | 72.38% |
| Precision | 71.71% | 69.57% |
| Recall | 68.55% | 80.50% |
| F1 | 70.10% | 74.64% |
| AUC | 76.5360% | 76.5360% |

The test-AUC 95% bootstrap interval is approximately **70.95% to 81.47%**.

The optimized threshold was chosen by maximizing F1 on training cross-validation out-of-fold predictions. It should not be described as a clinical cutoff.

## Saved artefacts

- Fitted model pipeline: [models/best_model.joblib](models/best_model.joblib)
- Model metadata: [models/best_model_meta.json](models/best_model_meta.json)
- Held-out test set: [data/processed/test_set.csv](data/processed/test_set.csv)
- Test results: [notebooks/OSMI_Analysis_Results/test_results.csv](notebooks/OSMI_Analysis_Results/test_results.csv)
- Cross-validation comparison: [notebooks/OSMI_Analysis_Results/model_comparison_cv.csv](notebooks/OSMI_Analysis_Results/model_comparison_cv.csv)
- Hyperparameter tuning: [notebooks/OSMI_Analysis_Results/tuning_results.csv](notebooks/OSMI_Analysis_Results/tuning_results.csv)

`models/best_model_without_work_interfere.joblib` is a legacy artifact from an earlier testing run. It is not the model selected by the final notebook and should not be used for Streamlit deployment.

## Streamlit application plan

The application will collect employee answers and produce a probability-based organizational support recommendation.

### Pending items before development

1. **Review the target definition**
   - Confirm that “treatment” means seeking mental-health treatment.
   - Confirm whether the question should be translated into a supportive recommendation or retained as a prediction label.

2. **Create the exact Streamlit input list**
   - Include the 22 features stored in `best_model_meta.json`.
   - Exclude the free-text comments field.
   - Exclude the `work_interfere` feature from the final application.

3. **Standardize input categories**
   - Add dropdowns for all categorical values.
   - Reuse the exact category names from the processed dataset.
   - Support missing or “prefer not to answer” choices where appropriate.

4. **Add a safe inference wrapper**
   - Load `best_model.joblib`.
   - Convert Streamlit answers into one-row model input.
   - Apply the saved 0.425 probability threshold.
   - Never expose the model as a diagnosis.

5. **Design the result page**
   - Show a probability and a confidence category.
   - Display a supportive message such as “Consider speaking with a counselor or reviewing available mental-health support.”
   - Include local counseling, HR, emergency, and crisis resources.

6. **Protect employee privacy**
   - Do not store names or answers by default.
   - Explain what data is retained, if anything.
   - Prefer in-session prediction without a database.
   - Complete organizational security and privacy approval.

7. **Address ethics and fairness**
   - Test performance across available demographic groups.
   - Avoid punitive or employment-related use.
   - State that the tool is advisory and voluntary.
   - Avoid screening or individual hiring decisions.

8. **Add model monitoring**
   - Record aggregate usage only, where approved.
   - Monitor changes in input distributions and prediction rates.
   - Retrain or suspend the model if performance deteriorates.

9. **Validate in a pilot**
   - Obtain HR and mental-health subject-matter expert approval.
   - Test usability with a small pilot group.
   - Verify that the language is supportive, accessible, and understandable.

10. **Prepare deployment**
    - Pin package versions.
    - Add Streamlit configuration and environment setup.
    - Define secret and access-control requirements.
    - Deploy to an approved internal environment.

## Recommended Streamlit MVP scope

The first version should contain:

- A short explanation and consent notice
- 22 form fields or a smaller pilot questionnaire agreed with stakeholders
- A model probability
- A low/medium/high support-recommendation category
- Mental-health resources
- A “start over” option
- No persistent employee-response storage

The first version should not include:

- Automatic diagnosis
- Employment decisions
- Individual HR reporting
- Saved employee profiles
- Emergency triage
- Claims of clinical accuracy

## Known limitations

- The survey is self-selected and specific to technology workers in 2014.
- The sample is relatively small and may not represent a current organization.
- Categorical definitions and workplace structures may not match current employees.
- Associations do not prove causation.
- A 0.425 threshold was tuned for F1, not clinical treatment appropriateness.
- The model is not suitable for diagnosis or individual medical decision-making.

## Reproducibility

Run the analysis in this order:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_model_training.ipynb`

The final modelling notebook creates or updates the model, metadata, test set, and analysis CSV files.

To execute all three notebooks in sequence from the project directory:

```bash
python notebooks/run_all_notebooks.py
```

The runner stops immediately and exits with status code 1 if a required notebook is missing or any notebook execution fails.

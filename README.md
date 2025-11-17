***ML-Zoomcamp Midterm Project***

# Annual Salary Prediction

[![Python](https://img.shields.io/badge/python-3.12.10-blue)](https://www.python.org/)

This project provides a machine learning solution to predict a candidate's annual salary based on publicly available profile information, assisting HR teams in making data-driven and informed compensation decisions.

## Problem Statement

In recruitment, HR professionals often encounter candidates with diverse profiles. After passing multiple interview rounds, candidates reach the salary negotiation stage. Many candidates choose not to disclose their previous salary, creating challenges:

- Offering too low a salary might cause the candidate to decline.
- Offering too high may exceed the company’s budget.

This project predicts a candidate’s annual salary using features such as department, job title, years of experience, location, performance rating, and more.


## Dataset

- **Source:** [Kaggle - HR Data MNC](https://www.kaggle.com/datasets/rohitgrewal/hr-data-mnc)

- **Download & Extract Dataset**:

```bash
# Download dataset from Kaggle
wget "https://www.kaggle.com/api/v1/datasets/download/rohitgrewal/hr-data-mnc"

# Unzip to folder 'hr_data'
unzip hr-data-mnc -d hr_data
```

- Original Size: 2,000,000 rows, 12 columns

- Columns: Unnamed: 0, Employee_ID, Full_Name, Department, Job_Title, Hire_Date, Location, Performance_Rating, Experience_Years, Status, Work_Mode, Salary_INR

- Target: Salary_INR

- Missing Values: None

## Data Preparation

- Reduced dataset to 600k rows for faster processing.
- Removed unique identifier columns: `Unnamed: 0`, `Employee_ID`, `Full_Name`.
- Extracted country from `Location` to reduce uniqueness.
- Converted `Hire_Date` to `Hire_Year` (only year). Later dropped due to high correlation with `Experience_Years`.
- Converted `Salary_INR` to local currency `Salary_VND`.
- Applied `np.log1p` on target variable due to long-tail distribution.
- Converted `Performance_Rating` to categorical type.
- Split dataset into training, validation, and test sets.
- Preprocessing and EDA scripts are located in `data_preparing_and_eda/`.

## Model Training

Tested four models with hyperparameter tuning:

| Model                     | Hyperparameters                                  | RMSE   | R²     | MAPE  |
|----------------------------|-------------------------------------------------|--------|--------|-------|
| LinearRegression           | Default                                         | 0.287  | 0.498  | 1.3%  |
| DecisionTreeRegressor      | max_depth=10, max_leaf_nodes=15, min_samples_leaf=4200 | 0.288  | 0.495  | 1.3%  |
| RandomForestRegressor      | n_estimators=45, max_depth=10, max_leaf_nodes=15, max_features=150 | 0.288  | 0.495  | 1.3%  |
| XGBoost                    | eta=0.3, max_depth=10, min_child_weight=1, num_boost_round=81, verbose_eval=5 | 0.289  | 0.491  | 1.3%  |

- Hyperparameter tuning details are in `training_model/`.
- All models perform similarly; **XGBoost is selected as the final model**.


## Deployment

### Required Libraries

- `pickle` — save model and DictVectorizer
- `fastapi` — create API
- `scikit-learn` — model training utilities
- `xgboost` — model training
- `uvicorn` — run API server
- `pydantic` — validate input JSON
- `requests` — test API

### Deployment Workflow

1. **Train and Save Model**
```bash
python final_train.py
```

2. **Create API**
```bash
python predict.py
```

3. **Test API**
```bash
python test.py
```

4. **Run App**
```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
uv run python test.py
```

## Virtual Environment

- Python version: 3.12.10
- Using `uv` to manage environment:

```bash
pip install uv
uv init
uv add scikit-learn fastapi xgboost pydantic uvicorn requests
uv sync --locked
```

## Containerization

- **Build Docker Image**

```bash
docker build -t predict-annual-salary .
```

- **Run Docker Container**
```bash
docker run -it --rm -p 9696:9696 predict-annual-salary
```

## Cloud Deployment

**Fly.io Deployment**
- Via website: select repo containing `Dockerfile` → deploy.
- Via CLI:

```bash
fly auth signup
fly launch --generate-name
fly deploy
```
- Remember to destroy the app after testing.
```bash
fly apps destroy <app-name>
```

from typing import Dict, Any
import pandas as pd
from matplotlib import pyplot as plt
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from openai import OpenAI
import random
from datetime import datetime, timedelta

FUNCTION_REGISTRY = {}

load_dotenv()
client = OpenAI(
    api_key = os.getenv("GROK_API_KEY"),
    base_url = os.getenv("GROK_API_URL"),
)

def register_function(name):
    def wrapper(func):
        FUNCTION_REGISTRY[name] = func
        return func
    return wrapper

def call_llm(description: str) -> Dict[str, str]:
    system_prompt = (
        "You are a machine learning assistant. Given a problem description, "
        "your job is to identify if it's a classification or regression problem "
        "and recommend the most suitable model accordingly."
    )
    
    user_prompt = (
        f"Problem Description: {description}\n"
        "Based on this, is it a classification or regression problem? "
        "Also, suggest a suitable machine learning model to solve it." \
        "And Respond in JSON format with the following keys:\n"
        "- recommended_model: The name of the model you recommend.\n"
        "- reason: A brief explanation of why you chose this model.\n"
    )
    
    completion = client.chat.completions.create(
        model="grok-3-latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    
    return {
        "response": completion.choices[0].message.content.strip()
    }

# Alternatively, you can use a mock function for local testing
def simulate_llm_model_selection(description: str) -> Dict[str, str]:
    if "predict" in description and "yes or no" in description.lower():
        return {
            "recommended_model": "Logistic Regression or Random Forest",
            "reason": "This sounds like a binary classification problem."
        }
    elif "group" in description or "cluster":
        return {
            "recommended_model": "KMeans or DBSCAN",
            "reason": "Unsupervised clustering is appropriate here."
        }
    elif "forecast" in description or "future":
        return {
            "recommended_model": "ARIMA or LSTM",
            "reason": "This is likely a time series prediction task."
        }
    else:
        return {
            "recommended_model": "Random Forest",
            "reason": "Default fallback for general problems."
        }


@register_function("plot_csv_chart")
def plot_csv_chart(payload: Dict[str, str], secrets: Dict[str, str], event_stream: list) -> Dict[str, Any]:
    try:
        csv_path = payload["csv_file"]
        x_col = payload["x_col"]
        y_col = payload["y_col"]
        df = pd.read_csv(csv_path)

        if x_col not in df.columns or y_col not in df.columns:
            return {"error": f"Columns '{x_col}' or '{y_col}' not found in CSV."}

        plt.figure(figsize=(10, 6))
        plt.plot(df[x_col], df[y_col], marker='o')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs {x_col}")
        plt.grid(True)
        output_path = "output_plot.png"
        plt.savefig(output_path)
        plt.close()

        return {
            "message": "Plot created successfully.",
            "output_file": output_path
        }

    except Exception as e:
        return {"error": str(e)}

@register_function("auto_clean_dataset")
def auto_clean_dataset(payload: Dict[str, str], secrets: Dict[str, str], event_stream: list) -> Dict[str, Any]:
    try:
        csv_path = payload["csv_file"]
        df = pd.read_csv(csv_path)
        thresh = len(df) * 0.7
        df.dropna(axis= 1,thresh=thresh)

        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = list(le.classes_)

        output_path = "cleaned_dataset.csv"
        df.to_csv(output_path, index=False)
        return {
            "message": "Dataset cleaned and saved successfully.",
            "output_file": output_path,
            "dropped_columns": list(set(payload.get("all_columns", [])) - set(df.columns)),
            "encoded_columns": list(label_encoders.keys())
        }

    except Exception as e:
        return {"error": str(e)}
    
@register_function("train_test_split_csv")
def train_test_split_csv(payload: Dict[str, Any], secrets: Dict[str, str], event_stream: list) -> Dict[str, Any]:
    try:
        csv_path = payload["csv_file"]
        features = payload["features"]           # List of feature column names
        target = payload["target"]               # Target column name
        test_size = payload.get("test_size", 0.2)

        df = pd.read_csv(csv_path)

        # Check column existence
        missing_cols = set(features + [target]) - set(df.columns)
        if missing_cols:
            return {"error": f"Missing columns: {missing_cols}"}

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=522)

        X_train.to_csv("X_train.csv", index=False)
        X_test.to_csv("X_test.csv", index=False)
        y_train.to_csv("y_train.csv", index=False)
        y_test.to_csv("y_test.csv", index=False)

        return {
            "message": "Train/test split completed.",
            "test_size": test_size,
            "output_files": ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
        }

    except Exception as e:
        return {"error": str(e)}
        
@register_function("select_model_llm")
def select_model_llm(payload: Dict[str, str], secrets: Dict[str, str], event_stream: list) -> Dict[str, Any]:
    try:
        description = payload.get("description")
        if not description:
            return {"error": "Missing 'description' in payload."}

        result = call_llm(description)
        return result

    except Exception as e:
        return {"error": str(e)}

@register_function("feature_summary")
def feature_summary(payload: Dict[str, str], secrets: Dict[str, str], event_stream: list) -> Dict[str, Any]:
    try:
        csv_path = payload["csv_file"]
        df = pd.read_csv(csv_path)

        summary = []

        for col in df.columns:
            col_summary = {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isnull().sum())
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                col_summary.update({
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                })
            else:
                col_summary["unique_values"] = int(df[col].nunique())

            summary.append(col_summary)

        pd.DataFrame(summary).to_csv("feature_summary.csv", index=False)

        return {
            "message": "Feature summary generated.",
            "summary": summary,
            "output_file": "feature_summary.csv"
        }

    except Exception as e:
        return {"error": str(e)}

@register_function("generate_synthetic_data")
def generate_synthetic_data(payload: Dict[str, Any], secrets: Dict[str, str], event_stream: list) -> Dict[str, Any]:
    """
    Generates synthetic data based on user-defined schema and saves it as CSV.
    """
    try:
        schema = payload["schema"]  # List of {"name": str, "type": str, "values": optional[list]}
        row_count = payload.get("row_count", 100)

        data = {}

        for col in schema:
            name = col["name"]
            col_type = col["type"]

            if col_type == "numeric":
                data[name] = [random.randint(10, 1000) for _ in range(row_count)]

            elif col_type == "categorical":
                values = col.get("values", ["A", "B", "C"])
                data[name] = [random.choice(values) for _ in range(row_count)]

            elif col_type == "date":
                start_date = datetime(2020, 1, 1)
                data[name] = [
                    (start_date + timedelta(days=random.randint(0, 1000))).strftime("%Y-%m-%d")
                    for _ in range(row_count)
                ]

            else:
                data[name] = ["unknown" for _ in range(row_count)]

        df = pd.DataFrame(data)
        output_file = "synthetic_data.csv"
        df.to_csv(output_file, index=False)

        return {
            "message": "Synthetic data generated successfully.",
            "output_file": output_file,
            "rows": row_count,
            "columns": list(data.keys())
        }

    except Exception as e:
        return {"error": str(e)}


from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import io
import base64
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

sns.set_theme(style="darkgrid")
plt.style.use('dark_background')

data_cache = {}

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#18181b', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

class PredictionInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@api_router.get("/data/info")
async def get_data_info():
    """
    Step 1: Load Dataset and Display Basic Information
    Returns dataset shape, columns, data types, and sample data
    """
    try:
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        
        data_cache['raw_df'] = df.copy()
        
        info = {
            "dataset_name": "California Housing Dataset",
            "description": "Real-world dataset with median house prices for California districts derived from the 1990 census",
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample_data": df.head(10).to_dict('records'),
            "target_variable": "MedHouseVal",
            "missing_values": df.isnull().sum().to_dict(),
            "total_missing": int(df.isnull().sum().sum())
        }
        
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/data/clean")
async def clean_data():
    """
    Step 2: Data Cleaning
    - Handle missing values (if any)
    - Detect and treat outliers using IQR method
    Returns cleaning statistics and steps
    """
    try:
        df = data_cache.get('raw_df')
        if df is None:
            raise HTTPException(status_code=400, detail="Please load data first using /data/info")
        
        df_clean = df.copy()
        cleaning_steps = []
        
        missing_before = df_clean.isnull().sum().sum()
        if missing_before > 0:
            df_clean = df_clean.fillna(df_clean.median())
            cleaning_steps.append({
                "step": "Handle Missing Values",
                "method": "Filled with median values",
                "count": int(missing_before)
            })
        else:
            cleaning_steps.append({
                "step": "Check Missing Values",
                "method": "No missing values found",
                "count": 0
            })
        
        outliers_info = {}
        for column in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_clean[(df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)]
            outliers_count = len(outliers)
            
            if outliers_count > 0:
                outliers_info[column] = {
                    "count": outliers_count,
                    "percentage": round((outliers_count / len(df_clean)) * 100, 2),
                    "lower_bound": round(lower_bound, 2),
                    "upper_bound": round(upper_bound, 2)
                }
                df_clean[column] = df_clean[column].clip(lower=lower_bound, upper=upper_bound)
        
        total_outliers = sum(info['count'] for info in outliers_info.values())
        cleaning_steps.append({
            "step": "Outlier Detection & Treatment",
            "method": "IQR Method (Interquartile Range) - Capped at 1.5*IQR",
            "count": total_outliers,
            "details": outliers_info
        })
        
        data_cache['clean_df'] = df_clean
        
        return {
            "original_shape": df.shape,
            "cleaned_shape": df_clean.shape,
            "cleaning_steps": cleaning_steps,
            "summary": f"Data cleaned successfully. Original: {df.shape[0]} rows, Clean: {df_clean.shape[0]} rows"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/data/eda")
async def perform_eda():
    """
    Step 3: Exploratory Data Analysis
    - Summary statistics
    - Correlation analysis
    - Distribution plots
    Returns statistics and base64 encoded plots
    """
    try:
        df = data_cache.get('clean_df')
        if df is None:
            raise HTTPException(status_code=400, detail="Please clean data first using /data/clean")
        
        summary_stats = df.describe().round(2).to_dict()
        
        correlation_matrix = df.corr().round(3).to_dict()
        
        fig1, ax1 = plt.subplots(figsize=(12, 10), facecolor='#18181b')
        ax1.set_facecolor('#18181b')
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, 
                    linewidths=1, linecolor='#27272a', fmt='.2f',
                    cbar_kws={'label': 'Correlation Coefficient'},
                    ax=ax1)
        ax1.set_title('Correlation Heatmap', color='#fafafa', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right', color='#fafafa')
        plt.yticks(rotation=0, color='#fafafa')
        correlation_plot = fig_to_base64(fig1)
        
        fig2, axes = plt.subplots(3, 3, figsize=(15, 12), facecolor='#18181b')
        fig2.suptitle('Feature Distributions', color='#fafafa', fontsize=18, y=0.995)
        axes = axes.flatten()
        
        for idx, column in enumerate(df.columns):
            axes[idx].set_facecolor('#18181b')
            axes[idx].hist(df[column], bins=50, color='#06b6d4', edgecolor='#22d3ee', alpha=0.7)
            axes[idx].set_title(column, color='#fafafa', fontsize=12)
            axes[idx].set_xlabel('Value', color='#a1a1aa', fontsize=10)
            axes[idx].set_ylabel('Frequency', color='#a1a1aa', fontsize=10)
            axes[idx].tick_params(colors='#a1a1aa')
            axes[idx].grid(True, alpha=0.2, color='#27272a')
        
        plt.tight_layout()
        distribution_plot = fig_to_base64(fig2)
        
        target_correlations = df.corr()['MedHouseVal'].sort_values(ascending=False).to_dict()
        
        return {
            "summary_statistics": summary_stats,
            "correlation_matrix": correlation_matrix,
            "target_correlations": target_correlations,
            "plots": {
                "correlation_heatmap": correlation_plot,
                "distributions": distribution_plot
            },
            "key_insights": [
                f"Dataset has {df.shape[0]} samples and {df.shape[1]} features",
                f"Target variable (MedHouseVal) has mean: ${df['MedHouseVal'].mean():.2f}K",
                f"Strongest positive correlation with target: {max(target_correlations, key=target_correlations.get)}",
                "No missing values in cleaned dataset"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/model/train")
async def train_model():
    """
    Step 4: Model Training
    - Split data into train/test sets (80/20)
    - Train Linear Regression model
    - Evaluate using RMSE, R², MAE
    Returns model performance metrics and visualizations
    """
    try:
        df = data_cache.get('clean_df')
        if df is None:
            raise HTTPException(status_code=400, detail="Please perform EDA first using /data/eda")
        
        X = df.drop('MedHouseVal', axis=1)
        y = df['MedHouseVal']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        lr_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, lr_pred))),
            "r2": float(r2_score(y_test, lr_pred)),
            "mae": float(mean_absolute_error(y_test, lr_pred))
        }
        
        rf_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, rf_pred))),
            "r2": float(r2_score(y_test, rf_pred)),
            "mae": float(mean_absolute_error(y_test, rf_pred))
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#18181b')
        
        axes[0].set_facecolor('#18181b')
        axes[0].scatter(y_test, lr_pred, alpha=0.5, color='#06b6d4', edgecolors='#22d3ee', s=30)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction', color='#ef4444')
        axes[0].set_xlabel('Actual Price ($100k)', color='#fafafa', fontsize=12)
        axes[0].set_ylabel('Predicted Price ($100k)', color='#fafafa', fontsize=12)
        axes[0].set_title(f'Linear Regression\nR² = {lr_metrics["r2"]:.3f}, RMSE = {lr_metrics["rmse"]:.3f}', 
                         color='#fafafa', fontsize=14)
        axes[0].legend(facecolor='#27272a', edgecolor='#06b6d4', labelcolor='#fafafa')
        axes[0].grid(True, alpha=0.2, color='#27272a')
        axes[0].tick_params(colors='#a1a1aa')
        
        axes[1].set_facecolor('#18181b')
        axes[1].scatter(y_test, rf_pred, alpha=0.5, color='#10b981', edgecolors='#22d3ee', s=30)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction', color='#ef4444')
        axes[1].set_xlabel('Actual Price ($100k)', color='#fafafa', fontsize=12)
        axes[1].set_ylabel('Predicted Price ($100k)', color='#fafafa', fontsize=12)
        axes[1].set_title(f'Random Forest\nR² = {rf_metrics["r2"]:.3f}, RMSE = {rf_metrics["rmse"]:.3f}', 
                         color='#fafafa', fontsize=14)
        axes[1].legend(facecolor='#27272a', edgecolor='#06b6d4', labelcolor='#fafafa')
        axes[1].grid(True, alpha=0.2, color='#27272a')
        axes[1].tick_params(colors='#a1a1aa')
        
        plt.tight_layout()
        predictions_plot = fig_to_base64(fig)
        
        feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
        sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        fig2, ax = plt.subplots(figsize=(10, 6), facecolor='#18181b')
        ax.set_facecolor('#18181b')
        features = list(sorted_features.keys())
        importances = list(sorted_features.values())
        ax.barh(features, importances, color='#06b6d4', edgecolor='#22d3ee')
        ax.set_xlabel('Feature Importance', color='#fafafa', fontsize=12)
        ax.set_title('Random Forest - Feature Importance', color='#fafafa', fontsize=14)
        ax.tick_params(colors='#a1a1aa')
        ax.grid(True, alpha=0.2, color='#27272a', axis='x')
        plt.tight_layout()
        importance_plot = fig_to_base64(fig2)
        
        data_cache['lr_model'] = lr_model
        data_cache['rf_model'] = rf_model
        data_cache['feature_columns'] = X.columns.tolist()
        
        return {
            "train_test_split": {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_percentage": 80,
                "test_percentage": 20
            },
            "models": {
                "linear_regression": lr_metrics,
                "random_forest": rf_metrics
            },
            "feature_importance": sorted_features,
            "plots": {
                "predictions": predictions_plot,
                "feature_importance": importance_plot
            },
            "conclusion": f"Random Forest performs better with R² of {rf_metrics['r2']:.3f} vs Linear Regression's {lr_metrics['r2']:.3f}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/model/predict")
async def make_prediction(input_data: PredictionInput):
    """
    Step 5: Make Predictions
    Use trained model to predict house prices for new data
    """
    try:
        rf_model = data_cache.get('rf_model')
        feature_columns = data_cache.get('feature_columns')
        
        if rf_model is None or feature_columns is None:
            raise HTTPException(status_code=400, detail="Please train model first using /model/train")
        
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict], columns=feature_columns)
        
        prediction = rf_model.predict(input_df)[0]
        
        return {
            "prediction": float(prediction),
            "prediction_formatted": f"${prediction * 100:.2f}K",
            "input_features": input_dict,
            "model_used": "Random Forest Regressor"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

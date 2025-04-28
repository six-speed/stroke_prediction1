from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from io import StringIO
import os
from fastapi.responses import RedirectResponse

# 读取模型
model = joblib.load("model/stroke_model.pkl")

# API 初始化
app = FastAPI(
    title="中风预测API",
    docs_url="/docs",
    openapi_url="/openapi.json"
)


class StrokeInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


@app.post("/predict_stroke")
def predict_stroke(data: StrokeInput):
    try:
        # 将输入数据转换为特征矩阵
        input_data = [[data.gender, data.age, data.hypertension, data.heart_disease,
                       data.ever_married, data.work_type, data.Residence_type,
                       data.avg_glucose_level, data.bmi, data.smoking_status]]
        input_df = pd.DataFrame(input_data, columns=["gender", "age", "hypertension", "heart_disease",
                                                    "ever_married", "work_type", "Residence_type",
                                                    "avg_glucose_level", "bmi", "smoking_status"])
        # 处理缺失值，与训练时保持一致
        input_df = input_df.fillna(0)
        # 对类别特征进行独热编码，与训练时保持一致
        categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
        input_df = pd.get_dummies(input_df, columns=categorical_cols)

        # 确保输入特征与训练时的特征一致，处理可能缺失的特征列
        train_cols = model.feature_names_in_
        for col in train_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[train_cols]

        X = input_df.values

        # 使用模型进行预测
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][int(pred)]

        # 根据概率得出风险评分
        risk = "高" if prob > 0.7 else ("中" if prob > 0.4 else "低")

        return {
            "预测中风": "是" if pred == 1 else "否",
            "中风概率": round(prob, 3),
            "风险评分": risk
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }

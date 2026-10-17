import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PATH, FEATURES, TARGET

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

df = pd.read_csv(DATA_PATH)

st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.subheader("Sample Data")
st.dataframe(df.head(10))

st.subheader("Class Distribution")
fig, ax = plt.subplots()
df[TARGET].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#3B8BD4", "#E85D24"])
ax.set_xticklabels(["Not Potable", "Potable"], rotation=0)
ax.set_ylabel("Count")
st.pyplot(fig)

st.subheader("Missing Values per Feature")
missing = df[FEATURES].isnull().sum()
st.bar_chart(missing)
st.caption("Missing values are filled with the training-set median (robust to skewed data).")

st.subheader("Feature Distributions")
feature = st.selectbox("Select Feature", FEATURES)
fig, ax = plt.subplots()
sns.histplot(df[feature], kde=True, ax=ax, color="#3B8BD4")
st.pyplot(fig)

st.subheader("Outliers (IQR method)")
q1, q3 = df[FEATURES].quantile(0.25), df[FEATURES].quantile(0.75)
iqr = q3 - q1
outliers = ((df[FEATURES] < q1 - 1.5 * iqr) | (df[FEATURES] > q3 + 1.5 * iqr)).sum()
st.dataframe(outliers.rename("Outlier count"))
st.caption("Outliers are detected but kept: the models are robust to moderate outliers.")

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

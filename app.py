import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Ozone Prediction App", layout="wide")

st.title("🌍 Ozone Prediction & Analysis Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("eighthr.data", header=None)

    df.columns = ["Date","WSR0","WSR1","WSR2","WSR3","WSR4","WSR5","WSR6","WSR7","WSR8","WSR9","WSR10","WSR11","WSR12","WSR13","WSR14","WSR15","WSR16","WSR17","WSR18","WSR19","WSR20","WSR21","WSR22","WSR23","WSR_PK","WSR_AV","T0","T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16","T17","T18","T19","T20","T21","T22","T23","T_PK","T_AV","T85","RH85","U85","V85","HT85","T70","RH70","U70","V70","HT70","T50","RH50","U50","V50","HT50","KI","TT","SLP","SLP_","Precp","Target"]

    # Cleaning
    df.replace('?', np.nan, inplace=True)
    df.iloc[:,1:73] = df.iloc[:,1:73].astype(float)

    df['Target'] = pd.Categorical(df['Target'], [0.0,1.0]).codes

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.set_index('Date')

    df = df.dropna()

    return df

df = load_data()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Controls")

show_data = st.sidebar.checkbox("Show Data")
show_corr = st.sidebar.checkbox("Show Correlation Heatmap")
show_plots = st.sidebar.checkbox("Show Visualizations")
run_model = st.sidebar.checkbox("Run ML Models")

# -------------------------------
# DATA DISPLAY
# -------------------------------
if show_data:
    st.subheader("Dataset")
    st.dataframe(df.head())

# -------------------------------
# CORRELATION
# -------------------------------
# if show_corr:
#     st.write(df.isnull().sum())
#     st.write(df.shape)
#     st.subheader("Correlation Heatmap")
#     fig, ax = plt.subplots(figsize=(12,6))
#     sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

# numeric_df = df.select_dtypes(include=np.number)

# # Remove constant columns
# numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

# fig, ax = plt.subplots(figsize=(12,6))
# sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=ax)

# st.pyplot(fig)
important_cols = [
    'WSR_AV','T_AV','T_PK','RH85','RH70',
    'HT85','HT70','SLP','Precp','Target'
]

fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(df[important_cols].corr(), annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig)

# -------------------------------
# VISUALIZATION
# -------------------------------
if show_plots:
    st.subheader("Visualizations")

    col1, col2 = st.columns(2)

    # Target Distribution
    with col1:
        st.write("Target Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Target'], kde=True, ax=ax)
        st.pyplot(fig)

    # Feature vs Target
    with col2:
        feature = st.selectbox(
            "Select Feature",
            ['T_AV','WSR_AV','SLP','RH85','Precp','KI','TT']
        )

        fig, ax = plt.subplots()
        ax.scatter(df[feature], df['Target'])
        ax.set_xlabel(feature)
        ax.set_ylabel("Target")
        st.pyplot(fig)

    # Time Series
    st.write("Target Over Time")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Target'])
    st.pyplot(fig)

# -------------------------------
# # MACHINE LEARNING
# # -------------------------------
# if run_model:
#     st.subheader("🤖 Machine Learning Models")

#     features = [
#         'WSR_PK','WSR_AV','T_PK','T_AV','T85','RH85','U85','V85','HT85',
#         'T70','RH70','U70','V70','HT70',
#         'T50','RH50','U50','V50','HT50',
#         'KI','TT','SLP','SLP_','Precp'
#     ]

#     X = df[features]
#     y = df['Target']

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Random Forest
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)

#     st.write("### Random Forest Report")
#     st.text(classification_report(y_test, y_pred))

#     # Decision Tree
#     clf = DecisionTreeClassifier(max_depth=10)
#     clf.fit(X_train, y_train)

#     score = clf.score(X_test, y_test)

#     st.write(f"### Decision Tree Accuracy: {score:.2f}")

# -------------------------------
# MACHINE LEARNING (PRO VERSION)
# -------------------------------
if run_model:
    st.subheader("🤖 Machine Learning Dashboard")

    features = [
        'WSR_PK','WSR_AV','T_PK','T_AV','T85','RH85','U85','V85','HT85',
        'T70','RH70','U70','V70','HT70',
        'T50','RH50','U50','V50','HT50',
        'KI','TT','SLP','SLP_','Precp'
    ]

    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Train Models
    # -------------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)

    dt = DecisionTreeClassifier(max_depth=10)
    dt.fit(X_train, y_train)
    dt_acc = dt.score(X_test, y_test)

    # -------------------------------
    # 📊 Show Metrics (Clean UI)
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🌲 Random Forest Accuracy", f"{rf_acc:.2f}")

    with col2:
        st.metric("🌳 Decision Tree Accuracy", f"{dt_acc:.2f}")

    # -------------------------------
    # 📈 Accuracy Comparison Chart
    # -------------------------------
    st.subheader("Model Comparison")

    model_df = pd.DataFrame({
        "Model": ["Random Forest", "Decision Tree"],
        "Accuracy": [rf_acc, dt_acc]
    })

    fig, ax = plt.subplots()
    ax.bar(model_df["Model"], model_df["Accuracy"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0,1)
    st.pyplot(fig)

    # -------------------------------
    # 📌 Feature Importance (VERY IMPORTANT)
    # -------------------------------
    st.subheader("📌 Feature Importance (Random Forest)")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(importance["Feature"], importance["Importance"])
    ax.invert_yaxis()
    st.pyplot(fig)

    # -------------------------------
    # 📋 Classification Report (Formatted)
    # -------------------------------
    st.subheader("📋 Classification Report")

    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # -------------------------------
    # 🎯 Prediction Section (BEST PART)
    # -------------------------------
    st.subheader("🎯 Make a Prediction")

    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        pred = rf.predict(input_df)[0]
        prob = rf.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"⚠️ High Ozone Level (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low Ozone Level (Probability: {prob:.2f})")
# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.write("Built using Streamlit 🚀")
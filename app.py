import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(page_title="Ozone Prediction App", layout="wide")

st.title("🌍 Ozone Prediction & Analysis Dashboard")
@st.cache_data
def load_data():
    df = pd.read_csv("eighthr.data", header=None)

    df.columns = ["Date","WSR0","WSR1","WSR2","WSR3","WSR4","WSR5","WSR6","WSR7","WSR8","WSR9","WSR10","WSR11","WSR12","WSR13","WSR14","WSR15","WSR16","WSR17","WSR18","WSR19","WSR20","WSR21","WSR22","WSR23","WSR_PK","WSR_AV","T0","T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16","T17","T18","T19","T20","T21","T22","T23","T_PK","T_AV","T85","RH85","U85","V85","HT85","T70","RH70","U70","V70","HT70","T50","RH50","U50","V50","HT50","KI","TT","SLP","SLP_","Precp","Target"]

    # Cleaning
    df.replace('?', np.nan, inplace=True)
    # df.iloc[:,1:73] = df.iloc[:,1:73].astype(float)
    # df.iloc[:,1:73] = df.iloc[:,1:73].apply(pd.to_numeric, errors='coerce')
    cols = df.columns[1:73]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    #df['Target'] = pd.Categorical(df['Target'], [0.0,1.0]).codes
    df['Target'] = pd.to_numeric(df['Target'], errors='coerce')
    df['Target'] = df['Target'].astype(int)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.set_index('Date')

    df = df.dropna()

    return df

df = load_data()
st.sidebar.header("Controls")

show_data = st.sidebar.checkbox("Show Data")
show_corr = st.sidebar.checkbox("Show Correlation Heatmap")
show_plots = st.sidebar.checkbox("Show Visualizations")
run_model = st.sidebar.checkbox("Run ML Models")
if show_data:
    st.subheader("Dataset")
    st.dataframe(df.head())

important_cols = [
    'WSR_AV','T_AV','T_PK','RH85','RH70',
    'HT85','HT70','SLP','Precp','Target'
]

fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(df[important_cols].corr(), annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig)
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

if run_model:
    st.subheader("🤖 Machine Learning Dashboard")

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, classification_report

    features = [
        'WSR_PK','WSR_AV','T_PK','T_AV','T85','RH85','U85','V85','HT85',
        'T70','RH70','U70','V70','HT70',
        'T50','RH50','U50','V50','HT50',
        'KI','TT','SLP','SLP_','Precp'
    ]

    X = df[features]
    y = df['Target']

    st.write("Class Distribution:", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:,1]

    rf_acc = rf.score(X_test, y_test)
    st.metric("🌲 Random Forest Accuracy", f"{rf_acc:.2f}")


    st.subheader("🎯 Adjust Prediction Sensitivity")
    threshold = st.slider("Set probability threshold", 0.0, 1.0, 0.3)

    st.subheader("🎯 Make a Prediction")

    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(
            f"{feature}", value=float(df[feature].mean())
        )

    input_df = pd.DataFrame([user_input])

    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        prob = rf.predict_proba(input_scaled)[0][1]

        if prob > threshold:
            st.error(f"⚠️ High Ozone Level (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low Ozone Level (Probability: {prob:.2f})")

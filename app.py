import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix
)

st.set_page_config(layout='wide')
st.title('Portopolio Saya')
st.header('Data Scientist')


# About Me
# ===============================
st.header("About Me")
st.write("""
**Hi, I’am Miftah Achmanda**  
I am an Informatics graduate from Telkom University with internship experience as an Academic Data Analyst at LLDIKTI IV and a Database Administrator at PT. Sisindokom Lintasbuana.  

I am passionate about Data Analysis, Data Science, and Business Analysis, with strong skills in Python, SQL, ETL, machine learning, Looker Studio, and Power BI.  

During my internship, I built interactive dashboards for monitoring 400+ universities and automated academic data processing, improving efficiency by 80% and data accuracy. I also contributed to Oracle database performance analysis and created more than 10 technical reports.  

I have completed several Data Science projects, including sentiment analysis using CNN/LSTM and automated data cleansing through API integration. I enjoy solving problems, working with large datasets, and creating clear data visualizations.  

My career goal is to grow in data-related roles—especially in the banking and tech sectors—by delivering impactful, data-driven solutions.
""")


# Page Config
# ===============================
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    layout="wide"
)

st.title("📊 Telco Customer Churn Prediction")
st.write("Prediksi churn pelanggan menggunakan Decision Tree & Random Forest")


# Load Data
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("1702184567307-WA_FnUseC_TelcoCustomerChurn.csv")
    return df

df = load_data()

st.subheader("🔍 Data Preview")
st.dataframe(df.head())


# Data Cleaning & Preparation
# ===============================
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)
df.drop('customerID', axis=1, inplace=True)


# Split Data
# ===============================
X = df.drop('Churn', axis=1)
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Feature Engineering
# ===============================
num_cols = x_train.select_dtypes(include=['int', 'float']).columns
cat_cols = x_train.select_dtypes(include='object').columns

scaler = StandardScaler()
encoder = OneHotEncoder(
    drop='first',
    sparse_output=False,
    handle_unknown='ignore'
)

x_train_scaled = pd.DataFrame(
    scaler.fit_transform(x_train[num_cols]),
    columns=num_cols,
    index=x_train.index
)

x_test_scaled = pd.DataFrame(
    scaler.transform(x_test[num_cols]),
    columns=num_cols,
    index=x_test.index
)

x_train_encoded = pd.DataFrame(
    encoder.fit_transform(x_train[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols),
    index=x_train.index
)

x_test_encoded = pd.DataFrame(
    encoder.transform(x_test[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols),
    index=x_test.index
)

x_train_final = pd.concat([x_train_scaled, x_train_encoded], axis=1)
x_test_final = pd.concat([x_test_scaled, x_test_encoded], axis=1)


# Modeling
# ===============================
dt_model = DecisionTreeClassifier(
    max_depth=6,
    random_state=123
)

rf_model = RandomForestClassifier(
    max_depth=10,
    n_estimators=200,
    random_state=123
)

dt_model.fit(x_train_final, y_train)
rf_model.fit(x_train_final, y_train)

# Evaluation Function
# ===============================
def evaluate_model(y_true, y_pred, y_prob, model_name):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob)
    }

# Predictions
# ===============================
dt_pred = dt_model.predict(x_test_final)
dt_prob = dt_model.predict_proba(x_test_final)[:, 1]

rf_pred = rf_model.predict(x_test_final)
rf_prob = rf_model.predict_proba(x_test_final)[:, 1]

results = pd.DataFrame([
    evaluate_model(y_test, dt_pred, dt_prob, "Decision Tree"),
    evaluate_model(y_test, rf_pred, rf_prob, "Random Forest")
])


# Model Evaluation Display
# ===============================
st.subheader("📈 Model Evaluation")

numeric_cols = results.select_dtypes(include='number').columns

st.dataframe(
    results.style.format({col: "{:.2f}" for col in numeric_cols})
)


# Confusion Matrix (RF)
# ===============================
st.subheader("🧩 Confusion Matrix – Random Forest")

cm = confusion_matrix(y_test, rf_pred)

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)


# Business Impact Simulation
# ===============================
st.subheader("💡 Potential Business Impact")
st.write("asumsi biaya retensi per pelanggan 500000")

total_churn = int(y_test.sum())
recall_rf = recall_score(y_test, rf_pred)
churn_detected = int(total_churn * recall_rf)

retention_cost = 500_000  # asumsi biaya retensi per pelanggan
potential_saving = churn_detected * retention_cost

col1, col2, col3 = st.columns(3)
col1.metric("Total Actual Churn", total_churn)
col2.metric("Churn Detected by Model", churn_detected)
col3.metric("Potential Cost Efficiency (Rp)", f"{potential_saving:,}")

st.success(
    """
    **Kesimpulan:**
    
    Random Forest dipilih sebagai model terbaik karena memiliki nilai AUC dan Precision
    yang lebih tinggi dibandingkan Decision Tree.
    
    Model ini lebih efektif untuk:
    - Mengurangi kehilangan pelanggan (Recall)
    - Efisiensi biaya retensi (Precision)
    - Menangani data tidak seimbang (AUC)
    """
)

st.subheader("📝 Prediksi Churn Customer Baru")

st.write("Masukkan data pelanggan untuk memprediksi kemungkinan churn:")

# ----------------------------
# Input form
# ----------------------------
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=12)
    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthlycharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    totalcharges = st.number_input("Total Charges", min_value=0.0, value=50.0)
    
    retention_cost_input = st.number_input("Biaya Retensi per Pelanggan (Rp)", min_value=0, value=500000)
    
    submitted = st.form_submit_button("Prediksi Churn")

# ----------------------------
# Prediksi
# ----------------------------
if submitted:
    # buat dataframe dari input user
    new_customer = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phoneservice],
        'MultipleLines': [multiplelines],
        'InternetService': [internet],
        'OnlineSecurity': [onlinesecurity],
        'OnlineBackup': [onlinebackup],
        'DeviceProtection': [deviceprotection],
        'TechSupport': [techsupport],
        'StreamingTV': [streamingtv],
        'StreamingMovies': [streamingmovies],
        'Contract': [contract],
        'PaperlessBilling': [paperless],
        'PaymentMethod': [payment],
        'MonthlyCharges': [monthlycharges],
        'TotalCharges': [totalcharges]
    })

    # ----------------------------
    # Preprocessing sama seperti dataset
    # ----------------------------
    # numeric
    new_num = pd.DataFrame(
        scaler.transform(new_customer[num_cols]),
        columns=num_cols
    )
    # categorical
    new_cat = pd.DataFrame(
        encoder.transform(new_customer[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols)
    )
    # gabungkan
    new_final = pd.concat([new_num, new_cat], axis=1)

    # prediksi
    pred_prob = rf_model.predict_proba(new_final)[:,1][0]
    pred_class = rf_model.predict(new_final)[0]

    # tampilkan hasil
    st.subheader("Hasil Prediksi")
    if pred_class == 1:
        st.warning(f"Pelanggan **Berpotensi Churn** dengan probabilitas {pred_prob*100:.2f}%")
        st.metric("Estimasi Biaya Retensi", f"Rp {int(pred_prob * retention_cost_input):,}")
    else:
        st.success(f"Pelanggan **Aman / Tidak Churn** dengan probabilitas {pred_prob*100:.2f}%")
        st.metric("Estimasi Biaya Retensi", f"Rp {retention_cost_input:,}")


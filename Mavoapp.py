import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')
st.set_page_config(page_title="CardioAI - Advanced Heart Disease Prediction", page_icon="❤️", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# 🎨 PROFESSIONAL CUSTOM CSS - MAKES IT LOOK STUNNING
# ============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white !important;
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card Styling */
    .css-1r6slb0 {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .css-1r6slb0:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border: 2px solid #e8ecff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #4a5568 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
        border-right: 1px solid #4a5568;
    }
    
    [data-testid="stSidebarNav"] {
        background-color: transparent;
    }
    
    .sidebar-content {
        color: white;
    }
    
    /* Success/Warning/Error Boxes */
    .success-box {
        background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%);
        border-left: 5px solid #48bb78;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #feebc8 0%, #fbd38d 100%);
        border-left: 5px solid #ed8936;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.2);
    }
    
    .error-box {
        background: linear-gradient(135deg, #fed7d7 0%, #fc8181 100%);
        border-left: 5px solid #f56565;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(245, 101, 101, 0.2);
    }
    
    /* Info Cards */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        transform: translateX(5px);
    }
    
    /* Section Headers */
    .section-title {
        color: #2d3748;
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Feature List */
    .feature-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 3px solid #48bb78;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 📦 LOAD MODELS
# ============================================================================
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    metrics = joblib.load("metrics.pkl")
    test_artifacts = joblib.load("test_artifacts.pkl")
    return rf_model, xgb_model, preprocessor, metrics, test_artifacts

rf_model, xgb_model, preprocessor, metrics, test_artifacts = load_artifacts()

# ============================================================================
# 🎨 SIDEBAR - PROFESSIONAL NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: white; font-size: 2.5rem; margin: 0;'>❤️ CardioAI</h1>
        <p style='color: #a0aec0; font-size: 0.9rem;'>Advanced Heart Disease Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard Overview", "📈 Model Performance", "🩺 New Patient Prediction", "🔍 Explainable AI (XAI)", "📖 Dataset & Features"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='color: #718096; font-size: 0.85rem; text-align: center; padding: 1rem;'>
        <p>Powered by Machine Learning & SHAP</p>
        <p>Built with ❤️ for Healthcare</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 🏠 PAGE 1: DASHBOARD OVERVIEW
# ============================================================================
if "🏠 Dashboard Overview" in page:
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>❤️ Welcome to CardioAI Dashboard</h1>
        <p>Advanced AI-Powered Heart Disease Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features Grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-card' style='border-left-color: #667eea;'>
            <h3 style='color: #667eea; margin-top: 0;'>🤖 Dual AI Models</h3>
            <p style='color: #4a5568; margin: 0;'>Choose between Random Forest and XGBoost for predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card' style='border-left-color: #48bb78;'>
            <h3 style='color: #48bb78; margin-top: 0;'>⚡ Real-Time Analysis</h3>
            <p style='color: #4a5568; margin: 0;'>Instant risk assessment with clinical recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-card' style='border-left-color: #ed8936;'>
            <h3 style='color: #ed8936; margin-top: 0;'>🔬 Explainable AI</h3>
            <p style='color: #4a5568; margin: 0;'>SHAP-based transparent decision making</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Stats Row
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Model Accuracy", "90.16%", "Excellent")
    with col2:
        st.metric("🎯 ROC-AUC Score", "0.964", "Outstanding")
    with col3:
        st.metric("👥 Patients Analyzed", "303", "Dataset")
    with col4:
        st.metric("🔬 Features Used", "13", "Clinical")
    
    # Welcome Message
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.08);'>
        <h2 class='section-title' style='margin-top: 0;'>🚀 What Can You Do?</h2>
        <div class='feature-item'>
            <strong>📈 View Model Performance:</strong> Analyze detailed metrics, ROC curves, and confusion matrices
        </div>
        <div class='feature-item'>
            <strong>🩺 Predict Patient Risk:</strong> Input clinical data to get instant heart disease risk assessment
        </div>
        <div class='feature-item'>
            <strong>🔍 Understand Predictions:</strong> Use SHAP explainability to see why the model made its decision
        </div>
        <div class='feature-item'>
            <strong>📖 Learn About Features:</strong> Understand the clinical significance of each predictor
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Image
    st.image("https://img.freepik.com/free-vector/heart-rate-monitor-concept-illustration_114360-1524.jpg", use_column_width=True)

# ============================================================================
# 📈 PAGE 2: MODEL PERFORMANCE
# ============================================================================
elif "📈 Model Performance" in page:
    st.markdown("""
    <div class='main-header'>
        <h1>📈 Model Performance Metrics</h1>
        <p>Comprehensive Evaluation of AI Models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Table
    st.markdown("<h2 class='section-title'>📊 Performance Comparison</h2>", unsafe_allow_html=True)
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.round(4)
    
    # Style the dataframe
    st.dataframe(
        metrics_df.style.background_gradient(cmap="Blues", subset=metrics_df.columns).format("{:.4f}"),
        use_container_width=True,
        height=300
    )
    
    # ROC Curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h2 class='section-title'>📈 ROC Curve Analysis</h2>", unsafe_allow_html=True)
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_predictions(test_artifacts["y_test"], test_artifacts["y_prob_rf"], name="Random Forest", ax=ax_roc, color="#667eea")
        RocCurveDisplay.from_predictions(test_artifacts["y_test"], test_artifacts["y_prob_xgb"], name="XGBoost", ax=ax_roc, color="#48bb78")
        ax_roc.plot([0, 1], [0, 1], "k--", label="Random Chance", alpha=0.5)
        ax_roc.set_title("Receiver Operating Characteristic", fontsize=14, fontweight='bold')
        ax_roc.set_xlabel("False Positive Rate", fontsize=12)
        ax_roc.set_ylabel("True Positive Rate", fontsize=12)
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, alpha=0.3)
        st.pyplot(fig_roc)
    
    with col2:
        st.markdown("<h2 class='section-title'>🎯 Confusion Matrix</h2>", unsafe_allow_html=True)
        model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
        
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        if model_choice == "Random Forest":
            ConfusionMatrixDisplay.from_predictions(test_artifacts["y_test"], test_artifacts["y_pred_rf"], ax=ax_cm, cmap="Blues")
        else:
            ConfusionMatrixDisplay.from_predictions(test_artifacts["y_test"], test_artifacts["y_pred_xgb"], ax=ax_cm, cmap="Greens")
        
        ax_cm.set_title(f"{model_choice} - Confusion Matrix", fontsize=14, fontweight='bold')
        st.pyplot(fig_cm)

# ============================================================================
# 🩺 PAGE 3: NEW PATIENT PREDICTION
# ============================================================================
elif "🩺 New Patient Prediction" in page:
    st.markdown("""
    <div class='main-header'>
        <h1>🩺 Patient Risk Assessment</h1>
        <p>Enter Clinical Data for AI-Powered Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<h3 style='color: #667eea;'>👤 Patient Demographics</h3>", unsafe_allow_html=True)
            age = st.number_input("Age (years)", min_value=20, max_value=100, value=50, help="Patient's age in years")
            sex = st.selectbox("Sex", ["Male", "Female"])
            chest_pain = st.selectbox("Chest Pain Type", ["typical", "nontypical", "nonanginal", "asymptomatic"])
            rest_bp = st.number_input("Resting BP (mmHg)", min_value=80, max_value=250, value=120)
        
        with col2:
            st.markdown("<h3 style='color: #48bb78;'>🔬 Laboratory Values</h3>", unsafe_allow_html=True)
            chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=500, value=200)
            fbs = st.selectbox("Fasting Blood Sugar >120", ["No", "Yes"])
            rest_ecg = st.selectbox("Resting ECG", [0, 1, 2], help="0=Normal, 1=ST-T abnormality, 2=LV hypertrophy")
            max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        
        with col3:
            st.markdown("<h3 style='color: #ed8936;'>🏥 Clinical Tests</h3>", unsafe_allow_html=True)
            ex_ang = st.selectbox("Exercise Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("ST Slope", [1, 2, 3], help="1=Upsloping, 2=Flat, 3=Downsloping")
            ca = st.number_input("Major Vessels (0-3)", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Thalassemia", ["normal", "fixed", "reversable"])
        
        st.markdown("---")
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            model_choice = st.selectbox("Select AI Model", ["Random Forest", "XGBoost"])
            submit = st.form_submit_button("🔮 Predict Risk Now", type="primary", use_container_width=True)
    
    if submit:
        input_data = pd.DataFrame({
            "Age": [age], "Sex": [1 if sex == "Male" else 0], "ChestPain": [chest_pain],
            "RestBP": [rest_bp], "Chol": [chol], "Fbs": [1 if fbs == "Yes" else 0],
            "RestECG": [rest_ecg], "MaxHR": [max_hr], "ExAng": [1 if ex_ang == "Yes" else 0],
            "Oldpeak": [oldpeak], "Slope": [slope], "Ca": [ca], "Thal": [thal]
        })
        
        feature_order = ["Age", "Sex", "ChestPain", "RestBP", "Chol", "Fbs", "RestECG", "MaxHR", "ExAng", "Oldpeak", "Slope", "Ca", "Thal"]
        input_data = input_data[feature_order]
        
        model = rf_model if model_choice == "Random Forest" else xgb_model
        pred_class = model.predict(input_data)[0]
        pred_prob = model.predict_proba(input_data)[0][1]
        
        st.session_state['latest_input'] = input_data
        st.session_state['latest_model'] = model
        st.session_state['latest_model_name'] = model_choice
        
        # Display Results
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h2 class='section-title'> Prediction Results</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if pred_class == 1:
                st.markdown(f"""
                <div class='error-box'>
                    <h3 style='margin-top: 0;'>🚨 Predicted Diagnosis</h3>
                    <h2 style='color: #f56565; margin: 0;'>Heart Disease Detected</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='success-box'>
                    <h3 style='margin-top: 0;'>✅ Predicted Diagnosis</h3>
                    <h2 style='color: #48bb78; margin: 0;'>No Heart Disease</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='info-card' style='border-left-color: #667eea;'>
                <h3 style='margin-top: 0;'>📊 Risk Probability</h3>
                <h2 style='color: #667eea; margin: 0;'>{pred_prob:.2%}</h2>
                <p style='color: #4a5568; margin: 0;'>Chance of Heart Disease</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Clinical Recommendation
        st.markdown("<h3 style='color: #2d3748;'>🩺 Clinical Recommendation</h3>", unsafe_allow_html=True)
        
        if pred_prob <= 0.30:
            st.markdown("""
            <div class='success-box'>
                <h3 style='margin-top: 0;'>🟢 Low Risk (0%-30%)</h3>
                <p style='margin: 0;'>Your risk of heart disease is low. Maintain a healthy lifestyle with regular exercise and a balanced diet. Continue regular check-ups.</p>
            </div>
            """, unsafe_allow_html=True)
        elif pred_prob <= 0.50:
            st.markdown("""
            <div class='warning-box'>
                <h3 style='margin-top: 0;'>🟡 Moderate Risk (31%-50%)</h3>
                <p style='margin: 0;'>You have a moderate risk of heart disease. Consider lifestyle modifications such as improving your diet, increasing physical activity, and monitoring your blood pressure and cholesterol.</p>
            </div>
            """, unsafe_allow_html=True)
        elif pred_prob <= 0.70:
            st.markdown("""
            <div class='error-box' style='background: linear-gradient(135deg, #fed7d7 0%, #fc8181 100%); border-left-color: #f56565;'>
                <h3 style='margin-top: 0;'>🟠 High Risk (51%-70%)</h3>
                <p style='margin: 0;'>You have a high risk of heart disease. It is strongly recommended to consult a healthcare provider for a thorough evaluation and possible preventive measures.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='error-box' style='background: linear-gradient(135deg, #feb2b2 0%, #fc8181 100%); border-left-color: #e53e3e;'>
                <h3 style='margin-top: 0;'>🔴 Very High Risk (71%-100%)</h3>
                <p style='margin: 0;'>You have a very high risk of heart disease. Immediate medical consultation is highly recommended for further diagnostic testing and intervention.</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# 🔍 PAGE 4: EXPLAINABLE AI
# ============================================================================
elif "🔍 Explainable AI (XAI)" in page:
    st.markdown("""
    <div class='main-header'>
        <h1>🔍 Explainable AI (XAI)</h1>
        <p>Understanding Model Decisions with SHAP</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'latest_input' in st.session_state:
        st.markdown("<h2 class='section-title'>🔎 Individual Prediction Explanation</h2>", unsafe_allow_html=True)
        
        input_data = st.session_state['latest_input']
        model = st.session_state['latest_model']
        model_name = st.session_state['latest_model_name']
        
        input_trans = preprocessor.transform(input_data)
        feature_names = preprocessor.get_feature_names_out()
        
        estimator = model.named_steps['xgb'] if model_name == "XGBoost" else model.named_steps['rf']
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(input_trans)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
        else:
            base_val = explainer.expected_value
        
        shap_exp = shap.Explanation(values=shap_values[0], base_values=float(base_val), data=input_trans[0], feature_names=feature_names)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(shap_exp, max_display=10, show=False)
        plt.title(f"SHAP Explanation - {model_name}", fontsize=16, fontweight='bold', pad=20)
        st.pyplot(fig)
    else:
        st.info("💡 Please make a prediction in the '🩺 New Patient Prediction' tab to see the local SHAP explanation.")
    
    st.markdown("---")
    st.markdown("<h2 class='section-title'>🌍 Global Feature Importance</h2>", unsafe_allow_html=True)
    
    x_test = test_artifacts["x_test"]
    x_test_trans = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()
    
    estimator_rf = rf_model.named_steps['rf']
    explainer_rf = shap.TreeExplainer(estimator_rf)
    shap_values_test = explainer_rf.shap_values(x_test_trans)
    
    if isinstance(shap_values_test, list):
        shap_values_test = shap_values_test[1]
    elif len(shap_values_test.shape) == 3:
        shap_values_test = shap_values_test[:, :, 1]
    
    base_val = explainer_rf.expected_value[1] if isinstance(explainer_rf.expected_value, (np.ndarray, list)) else explainer_rf.expected_value
    base_values_array = np.full(x_test_trans.shape[0], float(base_val))
    
    shap_exp_global = shap.Explanation(
        values=shap_values_test, 
        base_values=base_values_array, 
        data=x_test_trans, 
        feature_names=feature_names
    )
    
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    shap.plots.bar(shap_exp_global, max_display=10, show=False)
    plt.title("Global Feature Importance (SHAP)", fontsize=16, fontweight='bold', pad=20)
    st.pyplot(fig2)

# ============================================================================
# 📖 PAGE 5: DATASET & FEATURES
# ============================================================================
elif "📖 Dataset & Features" in page:
    st.markdown("""
    <div class='main-header'>
        <h1>📖 Dataset & Clinical Features</h1>
        <p>Understanding the Predictors</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.08);'>
        <p style='font-size: 1.1rem; color: #4a5568;'>This dashboard is built on a validated clinical dataset with 303 patients. Below is the clinical relevance of each feature used for prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    feature_info = {
        "Age": {"desc": "Patient's age in years", "relevance": "Strong predictor of cardiovascular risk", "icon": "👴"},
        "Sex": {"desc": "Biological sex (0=Female, 1=Male)", "relevance": "Men have higher baseline CVD risk", "icon": "👤"},
        "ChestPain": {"desc": "Type of chest pain", "relevance": "Asymptomatic often indicates severe disease", "icon": "💔"},
        "RestBP": {"desc": "Resting blood pressure (mmHg)", "relevance": "Hypertension is a major CVD risk factor", "icon": "🩺"},
        "Chol": {"desc": "Serum cholesterol (mg/dL)", "relevance": "Elevated LDL contributes to atherosclerosis", "icon": "🧪"},
        "Fbs": {"desc": "Fasting blood sugar >120 mg/dL", "relevance": "Indicator of diabetes/metabolic syndrome", "icon": "🔬"},
        "RestECG": {"desc": "Resting ECG results", "relevance": "Detects cardiac electrical abnormalities", "icon": "📊"},
        "MaxHR": {"desc": "Maximum heart rate (bpm)", "relevance": "Lower values may indicate poor cardiac function", "icon": "❤️"},
        "ExAng": {"desc": "Exercise-induced angina", "relevance": "Suggests ischemia during exertion", "icon": "🏃"},
        "Oldpeak": {"desc": "ST depression by exercise", "relevance": "Marker of myocardial ischemia", "icon": "📉"},
        "Slope": {"desc": "Peak exercise ST slope", "relevance": "Downsloping = higher risk", "icon": "📐"},
        "Ca": {"desc": "Number of major vessels (0-3)", "relevance": "More vessels = more severe disease", "icon": ""},
        "Thal": {"desc": "Thalassemia stress test", "relevance": "Defects indicate ischemia", "icon": "🧬"}
    }
    
    for feature, info in feature_info.items():
        st.markdown(f"""
        <div class='info-card'>
            <h3 style='color: #667eea; margin-top: 0;'>{info['icon']} {feature}</h3>
            <p style='color: #4a5568; margin: 0.5rem 0;'><strong>Description:</strong> {info['desc']}</p>
            <p style='color: #48bb78; margin: 0;'><strong>Clinical Relevance:</strong> {info['relevance']}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #718096; font-size: 0.9rem;'>
    <p>CardioAI - Advanced Heart Disease Prediction System</p>
    <p>Built with Streamlit ❤️ | Powered by Machine Learning & SHAP</p>
</div>
""", unsafe_allow_html=True)
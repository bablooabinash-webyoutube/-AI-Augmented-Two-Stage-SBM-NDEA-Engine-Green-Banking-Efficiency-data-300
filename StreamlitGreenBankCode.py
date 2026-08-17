"""
===============================================================================
AI-AUGMENTED TWO-STAGE SBM-NDEA ENGINE: GREEN BANKING EFFICIENCY
===============================================================================
Streamlit Enterprise Interactive Dashboard & Decision Support System
===============================================================================
"""
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & STYLES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Green Bank AI Engine", 
    page_icon="🌱", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for metric cards and typography
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
    .card-title { font-size: 0.9rem; font-weight: 600; color: #6B7280; }
    .card-value { font-size: 1.8rem; font-weight: 700; color: #111827; }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & ML SURROGATE CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('bank_efficiency_data.csv')
    # Standardize archetype labels for UI compatibility
    archetype_map = {
        'Archetype_A': 'Efficient',
        'Archetype_B': 'Stage 1 Bottleneck',
        'Archetype_C': 'Stage 2 Bottleneck'
    }
    df['True_Archetype'] = df['True_Archetype'].replace(archetype_map)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Data file 'bank_efficiency_data.csv' not found. Please place 'bank_efficiency_data.csv' in the exact same directory as this script.")
    st.stop()

@st.cache_resource
def load_trained_models():
    """Loads the pre-trained frozen .joblib models instead of training live."""
    surrogate_reg = joblib.load("sbm_rf_model.joblib")
    mlp_reg = joblib.load("sbm_mlp_model.joblib")
    scaler = joblib.load("sbm_scaler.joblib")

    # Extract feature importances for the XAI Tab
    feature_importances = surrogate_reg.feature_importances_

    return surrogate_reg, mlp_reg, scaler, feature_importances

surrogate_reg, mlp_reg, scaler, feature_importances = load_trained_models()

@st.cache_resource
def train_missing_classifier(data):
    """Trains a lightweight classifier for the Archetype predictions in Tabs 3 & 4."""
    feature_cols = ['x1_OpCost', 'x2_Staff', 'z1_GreenDeposits', 'z2_ESGAccounts', 'y1_GreenRevenue', 'y2_CarbonNPL']
    X = data[feature_cols].values
    y_cls = data['True_Archetype'].values
    return RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y_cls)

surrogate_clf = train_missing_classifier(df)

# -----------------------------------------------------------------------------
# 3. SIDEBAR FILTERS & CONTROL PANEL
# -----------------------------------------------------------------------------
st.sidebar.title("🏦 Operational Control Panel")
st.sidebar.markdown("---")

branch_list = df['Branch_ID'].tolist()
selected_branch = st.sidebar.selectbox("🔍 Select Branch to Audit:", branch_list)
branch_data = df[df['Branch_ID'] == selected_branch].iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Model Orientation View")
orientation_view = st.sidebar.radio(
    "Choose Metric Orientation:",
    ["Non-Oriented (Charnes-Cooper)", "Input-Oriented (Cost/Risk)", "Output-Oriented (Revenue/Growth)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Architecture Summary")
st.sidebar.info(
    "This platform combines **Two-Stage SBM-NDEA Optimization** "
    "with a **Deep Learning Surrogate Layer** for real-time operational triage."
)

# Map orientation selection to dataframe columns
if "Input" in orientation_view:
    e0_col, e1_col, e2_col = 'E0_Input', 'E1_Input', 'E2_Input'
elif "Output" in orientation_view:
    e0_col, e1_col, e2_col = 'E0_Output', 'E1_Output', 'E2_Output'
else:
    e0_col, e1_col, e2_col = 'E0_NonOriented', 'E1_NonOriented', 'E2_NonOriented'

# -----------------------------------------------------------------------------
# 4. HEADER & NAVIGATION TABS
# -----------------------------------------------------------------------------
st.markdown("<p class='main-header'>🌱 AI-Augmented Green Banking Engine</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Real-Time Two-Stage Network SBM-NDEA Benchmarking & Prescriptive Analytics</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Branch Audit & Prescriptive Targets", 
    "📊 Network Insights & XAI Drivers", 
    "⚡ Real-Time AI Scenario Simulator",
    "🩺 Custom Bank Health Audit"
])

# =============================================================================
# TAB 1: INDIVIDUAL BRANCH AUDIT, TARGET ADJUSTMENTS & ACTION PLAN
# =============================================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)

    eff_e0 = branch_data[e0_col] * 100
    eff_e1 = branch_data[e1_col] * 100
    eff_e2 = branch_data[e2_col] * 100
    archetype = branch_data['True_Archetype']

    avg_net_eff = df[e0_col].mean() * 100
    delta_eff = eff_e0 - avg_net_eff

    c1.metric("Overall SBM Efficiency ($E_0$)", f"{eff_e0:.1f}%", delta=f"{delta_eff:+.1f}% vs Avg")
    c2.metric("Stage 1 Efficiency ($E_1$)", f"{eff_e1:.1f}%", help="Green Conversion & Operations")
    c3.metric("Stage 2 Efficiency ($E_2$)", f"{eff_e2:.1f}%", help="Sustainable Profitability & Risk")
    c4.metric("Operational Archetype", archetype)

    st.markdown("---")
    st.subheader(f"🎯 Target Adjustments to Reach 100% SBM Efficiency ({selected_branch})")

    if archetype == 'Efficient':
        st.success("🌟 **Benchmark Branch:** Operating on the efficient frontier! No input cuts or output adjustments required.")
    else:
        st.markdown(
            "To reach **100% operational efficiency ($E_0 = 1.0$)**, the SBM optimization engine recommends "
            "the following exact input reductions and output expansions:"
        )

        curr_opcost = branch_data['x1_OpCost']
        target_opcost = curr_opcost * (1 - branch_data['Reduct_%_OpCost'] / 100)
        curr_staff = branch_data['x2_Staff']
        target_staff = curr_staff * (1 - branch_data['Reduct_%_Staff'] / 100)
        curr_rev = branch_data['y1_GreenRevenue']
        target_rev = curr_rev * (1 + branch_data['Expand_%_Revenue'] / 100)
        curr_npl = branch_data['y2_CarbonNPL']
        target_npl = curr_npl * (1 - branch_data['Mitigat_%_CarbonNPL'] / 100)

        target_data = [
            {"Variable Category": "Stage 1 Input", "Metric Name": "Operational Cost ($M)", "Action Required": "🔻 Decrease Input", "Current Value": f"${curr_opcost:.2f}M", "Recommended Target": f"${target_opcost:.2f}M", "Adjustment": f"-{branch_data['Reduct_%_OpCost']:.1f}%"},
            {"Variable Category": "Stage 1 Input", "Metric Name": "Branch Staffing (FTE)", "Action Required": "🔻 Decrease Input", "Current Value": f"{int(curr_staff)} FTEs", "Recommended Target": f"{int(np.ceil(target_staff))} FTEs", "Adjustment": f"-{branch_data['Reduct_%_Staff']:.1f}%"},
            {"Variable Category": "Stage 2 Output", "Metric Name": "Green Revenue ($M)", "Action Required": "🔺 Increase Output", "Current Value": f"${curr_rev:.2f}M", "Recommended Target": f"${target_rev:.2f}M", "Adjustment": f"+{branch_data['Expand_%_Revenue']:.1f}%"},
            {"Variable Category": "Stage 2 Risk Output", "Metric Name": "Carbon NPLs ($M)", "Action Required": "🔻 Mitigate Risk", "Current Value": f"${curr_npl:.2f}M", "Recommended Target": f"${target_npl:.2f}M", "Adjustment": f"-{branch_data['Mitigat_%_CarbonNPL']:.1f}%"}
        ]

        target_df = pd.DataFrame(target_data)
        st.dataframe(target_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    v_left, v_right = st.columns(2)

    with v_left:
        st.subheader("📍 Network Efficiency Positioning")
        fig_scatter = px.scatter(
            df, 
            x='x1_OpCost', 
            y='y1_GreenRevenue', 
            color='True_Archetype',
            size=e0_col,
            hover_data=['Branch_ID', e0_col],
            title="Op Cost vs. Green Revenue (Bubble Size = Efficiency)",
            labels={'x1_OpCost': 'Op Cost ($M)', 'y1_GreenRevenue': 'Green Revenue ($M)'},
            color_discrete_map={'Efficient': '#2ca02c', 'Stage 1 Bottleneck': '#ff7f0e', 'Stage 2 Bottleneck': '#d62728'}
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[branch_data['x1_OpCost']], y=[branch_data['y1_GreenRevenue']],
                mode='markers', marker=dict(size=20, color='yellow', symbol='star', line=dict(width=2, color='black')),
                name=f"Selected ({selected_branch})"
            )
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with v_right:
        st.subheader("📊 Required Adjustments Breakdown")
        if archetype != 'Efficient':
            targets = {'OpCost Cut (%)': branch_data['Reduct_%_OpCost'], 'Staff Cut (%)': branch_data['Reduct_%_Staff'], 'Revenue Boost (%)': branch_data['Expand_%_Revenue'], 'Carbon NPL Mitig. (%)': branch_data['Mitigat_%_CarbonNPL']}
            target_chart_df = pd.DataFrame(list(targets.items()), columns=['Metric', 'Adjustment (%)'])
            fig_bar = px.bar(target_chart_df, x='Metric', y='Adjustment (%)', color='Metric', text_auto='.1f', title=f"Slack Improvement Targets ({archetype})")
            fig_bar.update_layout(showlegend=False, yaxis_title="Adjustment Needed (%)")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No charts displayed for efficient branches as no adjustments are required.")

# =============================================================================
# TAB 2: NETWORK-WIDE BENCHMARKING & EXPLAINABLE AI (XAI)
# =============================================================================
with tab2:
    st.subheader("📊 Network-Wide Archetype & Efficiency Distribution")

    n1, n2 = st.columns(2)
    with n1:
        arch_counts = df['True_Archetype'].value_counts().reset_index()
        arch_counts.columns = ['Archetype', 'Count']
        fig_pie = px.pie(arch_counts, values='Count', names='Archetype', title='Branch Archetype Breakdown', color='Archetype', color_discrete_map={'Efficient': '#2ca02c', 'Stage 1 Bottleneck': '#ff7f0e', 'Stage 2 Bottleneck': '#d62728'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with n2:
        fig_box = px.box(df, x='True_Archetype', y=['E1_NonOriented', 'E2_NonOriented'], title="Stage 1 ($E_1$) vs. Stage 2 ($E_2$) Efficiency Spread", labels={'value': 'Efficiency Score', 'variable': 'Stage'})
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Explainable AI (XAI): Global Feature Importance")

    xai_df = pd.DataFrame({
        'Operational Feature': ['x1_OpCost', 'x2_Staff', 'z1_GreenDeposits', 'z2_ESGAccounts', 'y1_GreenRevenue', 'y2_CarbonNPL'],
        'Importance Weight': feature_importances
    }).sort_values(by='Importance Weight', ascending=True)

    fig_xai = px.bar(xai_df, x='Importance Weight', y='Operational Feature', orientation='h', title="Random Forest Global Feature Drivers (Network-Wide SBM Impact)", text_auto='.3f', color='Importance Weight', color_continuous_scale='Greens')
    st.plotly_chart(fig_xai, use_container_width=True)

    st.markdown("---")
    st.subheader("📑 Raw SBM Network Audit Table")
    st.dataframe(df[['Branch_ID', 'True_Archetype', 'E0_NonOriented', 'E1_NonOriented', 'E2_NonOriented', 'Reduct_%_OpCost', 'Expand_%_Revenue', 'Mitigat_%_CarbonNPL']], use_container_width=True)

# =============================================================================
# TAB 3: REAL-TIME "WHAT-IF" AI SURROGATE SIMULATOR
# =============================================================================
with tab3:
    st.subheader("⚡ Live Operational Scenario Simulator")
    st.markdown("Simulate hypothetical operational modifications for a branch and obtain instant AI surrogate predictions (<1ms inference).")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("##### 📥 Stage 1 Inputs")
        sim_x1 = st.slider("Op Cost ($M)", 0.5, 7.0, float(branch_data['x1_OpCost']), 0.1)
        sim_x2 = st.slider("Staff Count", 5, 60, int(branch_data['x2_Staff']), 1)

    with s2:
        st.markdown("##### 🔄 Intermediates")
        sim_z1 = st.slider("Green Deposits ($M)", 5.0, 40.0, float(branch_data['z1_GreenDeposits']), 0.5)
        sim_z2 = st.slider("ESG Accounts", 100, 1200, int(branch_data['z2_ESGAccounts']), 10)

    with s3:
        st.markdown("##### 📤 Stage 2 Outputs & Risk")
        sim_y1 = st.slider("Green Revenue ($M)", 1.0, 10.0, float(branch_data['y1_GreenRevenue']), 0.1)
        sim_y2 = st.slider("Carbon NPL ($M)", 0.05, 6.0, float(branch_data['y2_CarbonNPL']), 0.05)

    input_vector = np.array([[sim_x1, sim_x2, sim_z1, sim_z2, sim_y1, sim_y2]])

    pred_archetype = surrogate_clf.predict(input_vector)[0]
    pred_efficiency = surrogate_reg.predict(input_vector)[0] * 100

    st.markdown("---")
    st.markdown("### 🔮 AI Surrogate Prediction Results")

    p1, p2 = st.columns(2)
    p1.metric("Predicted SBM Efficiency Score", f"{pred_efficiency:.1f}%")
    p2.metric("Predicted Operational Archetype", pred_archetype)

    if pred_archetype == 'Efficient':
        st.success("✅ **Scenario Outcome:** The branch reaches the efficient frontier under these parameters!")
    elif pred_archetype == 'Stage 1 Bottleneck':
        st.warning("⚠️ **Scenario Outcome:** Operational waste detected in front-office inputs.")
    else:
        st.error("🚨 **Scenario Outcome:** High credit risk or poor revenue conversion detected in Stage 2.")

# =============================================================================
# TAB 4: CUSTOM BANK & BRANCH HEALTH AUDIT
# =============================================================================
with tab4:
    st.subheader("🩺 Custom Bank & Branch Health Diagnostic Report")
    st.markdown("Designed specifically for **Bank Managers & Regional Authorities**. Enter custom operational numbers below to generate an instant SBM Efficiency score, risk diagnostic, and an executive health report.")

    st.markdown("---")

    with st.form("custom_bank_form"):
        st.markdown("#### 📝 Step 1: Input Bank / Branch Operational Data")

        b_name = st.text_input("Branch / Institution Name", value="Custom Regional Branch")

        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            st.markdown("##### 📥 Stage 1 Inputs (Resources)")
            c_x1 = st.number_input("Operational Cost ($M)", min_value=0.1, max_value=50.0, value=3.2, step=0.1)
            c_x2 = st.number_input("Staff Count (FTE)", min_value=1, max_value=500, value=25, step=1)

        with col_in2:
            st.markdown("##### 🔄 Intermediates (Green Accounts)")
            c_z1 = st.number_input("Green Deposits ($M)", min_value=0.1, max_value=200.0, value=18.5, step=0.5)
            c_z2 = st.number_input("ESG Accounts Active", min_value=10, max_value=10000, value=450, step=10)

        with col_in3:
            st.markdown("##### 📤 Stage 2 Outputs & Risk")
            c_y1 = st.number_input("Green Revenue Generated ($M)", min_value=0.1, max_value=100.0, value=4.2, step=0.1)
            c_y2 = st.number_input("Carbon NPL ($M)", min_value=0.0, max_value=50.0, value=0.8, step=0.05)

        submit_btn = st.form_submit_button("🚀 Run Health Audit & Generate Executive Report", use_container_width=True)

    if submit_btn or 'custom_run' in st.session_state:
        st.session_state['custom_run'] = True

        custom_vec = np.array([[c_x1, c_x2, c_z1, c_z2, c_y1, c_y2]])
        pred_arch = surrogate_clf.predict(custom_vec)[0]
        pred_eff = float(surrogate_reg.predict(custom_vec)[0] * 100)

        if pred_eff >= 95.0:
            grade, grade_badge = "A+ (Optimal Frontier)", "🟢"
        elif pred_eff >= 80.0:
            grade, grade_badge = "B (Moderate Overhead)", "🟡"
        elif pred_eff >= 65.0:
            grade, grade_badge = "C (Significant Inefficiency)", "🟠"
        else:
            grade, grade_badge = "D (High Operational Risk)", "🔴"

        st.markdown("---")
        st.markdown(f"## 📊 Executive Health Report: **{b_name}**")

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Predicted Efficiency Score", f"{pred_eff:.1f}%")
        h2.metric("Health Rating", f"{grade_badge} {grade}")
        h3.metric("Operational Archetype", pred_arch)

        net_avg_eff = df['E0_NonOriented'].mean() * 100
        diff_net = pred_eff - net_avg_eff
        h4.metric("Vs. Network Average", f"{diff_net:+.1f}%", delta=f"{diff_net:+.1f}%")

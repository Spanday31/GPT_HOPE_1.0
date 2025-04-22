import streamlit as st
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
from datetime import datetime
from dataclasses import dataclass

# ======================
# CONSTANTS
# ======================

LDL_THERAPIES = {
    "Atorvastatin 20 mg": {"reduction": 40},
    "Atorvastatin 80 mg": {"reduction": 50},
    "Rosuvastatin 10 mg": {"reduction": 45},
    "Rosuvastatin 20 mg": {"reduction": 55}
}

# ======================
# DATA CLASSES
# ======================

@dataclass
class PatientData:
    name: str
    age: int
    sex: str

@dataclass
class RiskData:
    baseline_risk: float
    final_risk: float
    current_ldl: float
    ldl_target: float
    recommendations: str

# ======================
# CALCULATION FUNCTIONS
# ======================

@st.cache_data
def calculate_smart_risk(age, sex, sbp, total_chol, hdl, smoker, diabetes, egfr, crp, vasc_count):
    try:
        sex_val = 1 if sex == "Male" else 0
        smoking_val = 1 if smoker else 0
        diabetes_val = 1 if diabetes else 0
        crp_log = math.log(max(crp, 0.1) + 1)
        lp = (0.064 * age + 0.34 * sex_val + 0.02 * sbp + 0.25 * total_chol -
              0.25 * hdl + 0.44 * smoking_val + 0.51 * diabetes_val -
              0.2 * (egfr / 10) + 0.25 * crp_log + 0.4 * vasc_count)
        risk10 = 1 - 0.900 ** math.exp(lp - 5.8)
        return max(1.0, min(99.0, round(risk10 * 100, 1)))
    except Exception as e:
        st.error(f"Error calculating risk: {str(e)}")
        return None

def calculate_ldl_effect(baseline_risk, baseline_ldl, final_ldl, bp_controlled=False, smoking_cessation=False, lifestyle=True):
    try:
        ldl_reduction = baseline_ldl - final_ldl
        rrr = min(22 * ldl_reduction, 60)
        if bp_controlled:
            rrr += 10
        if smoking_cessation:
            rrr += 10
        if lifestyle:
            rrr += 5
        rrr = min(rrr, 80)
        return baseline_risk * (1 - rrr / 100)
    except Exception as e:
        st.error(f"Error calculating LDL effect: {str(e)}")
        return baseline_risk

def calculate_ldl_reduction(current_ldl, pre_statin, discharge_statin, discharge_add_ons):
    statin_reduction = LDL_THERAPIES.get(discharge_statin, {}).get("reduction", 0)
    if pre_statin != "None":
        statin_reduction *= 0.5
    total_reduction = statin_reduction
    if "Ezetimibe" in discharge_add_ons:
        total_reduction += 20
    if "PCSK9 inhibitor" in discharge_add_ons:
        total_reduction += 60
    if "Inclisiran" in discharge_add_ons:
        total_reduction += 50
    projected_ldl = current_ldl * (1 - total_reduction / 100)
    return projected_ldl, total_reduction

def generate_recommendations(final_risk):
    if final_risk >= 30:
        return "🔴 Very High Risk: High-intensity statin, PCSK9 inhibitor, SBP <130 mmHg, lifestyle adherence."
    elif final_risk >= 20:
        return "🟠 High Risk: Moderate-intensity statin, SBP <130 mmHg, encourage smoking cessation and healthy diet."
    else:
        return "🟢 Moderate Risk: Lifestyle adherence, annual reassessment."

# ======================
# VISUALIZATION
# ======================

def plot_risk_gauge(risk_value):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 10)
    ax.axis('off')

    bands = [(0, 20, 'green'), (20, 30, 'orange'), (30, 100, 'red')]
    for start, end, color in bands:
        ax.barh(5, np.radians(end - start), left=np.radians(start), height=5, color=color, alpha=0.7)

    needle_angle = np.radians(min(max(risk_value, 0), 99))
    ax.annotate('', xy=(needle_angle, 7), xytext=(0, 0),
                arrowprops=dict(facecolor='black', width=2, headwidth=8))

    ax.text(0, -1.5, f"Risk: {risk_value:.1f}%", ha='center', va='center', fontsize=14, fontweight='bold')

    return fig

# ======================
# STREAMLIT APP SETUP
# ======================

st.set_page_config(page_title="PRIME CVD Risk Calculator", layout="wide", page_icon="❤️")

st.title("PRIME CVD Risk Calculator")

st.sidebar.header("Patient Demographics")
age = st.sidebar.number_input("Age", 30, 100, 65)
sex = st.sidebar.radio("Sex", ["Male", "Female"], horizontal=True)
diabetes = st.sidebar.checkbox("Diabetes mellitus")
smoker = st.sidebar.checkbox("Current smoker")

st.sidebar.header("Known Vascular Disease Territory")
cad = st.sidebar.checkbox("Coronary artery disease")
stroke = st.sidebar.checkbox("Cerebrovascular disease")
pad = st.sidebar.checkbox("Peripheral artery disease")
vasc_count = sum([cad, stroke, pad])

st.sidebar.header("Biomarkers")
total_chol = st.sidebar.number_input("Total Cholesterol (mmol/L)", 2.0, 10.0, 5.0, 0.1)
hdl = st.sidebar.number_input("HDL-C (mmol/L)", 0.5, 3.0, 1.0, 0.1)
ldl = st.sidebar.number_input("LDL-C (mmol/L)", 0.5, 6.0, 3.5, 0.1)
sbp = st.sidebar.number_input("SBP (mmHg)", 90, 220, 140)
egfr = st.sidebar.slider("eGFR (mL/min/1.73m²)", 15, 120, 80)
crp = st.sidebar.number_input("hs-CRP (mg/L) (not during acute illness)", 0.1, 20.0, 2.0, 0.1)
include_crp = st.sidebar.checkbox("Include hs-CRP in risk calculation", value=True)

st.sidebar.header("Other Modifiable Factors")
bp_controlled = st.sidebar.checkbox("Blood pressure controlled <130/80")
smoking_cessation = st.sidebar.checkbox("Smoking cessation planned/achieved")
lifestyle = st.sidebar.checkbox("Healthy diet & regular exercise", value=True)

# ======================
# MAIN LOGIC
# ======================

if hdl >= total_chol:
    st.warning("HDL should be lower than total cholesterol for accurate calculation.")

crp_used = crp if include_crp else 0.0
baseline_risk = calculate_smart_risk(age, sex, sbp, total_chol, hdl, smoker, diabetes, egfr, crp_used, vasc_count)

if baseline_risk:
    st.success(f"Baseline 10-Year Risk: {baseline_risk}%")

    with st.form("treatment_form"):
        pre_statin = st.selectbox("Current Statin", ["None"] + list(LDL_THERAPIES.keys()), index=0)
        discharge_statin = st.selectbox("Recommended Statin", ["None"] + list(LDL_THERAPIES.keys()), index=2)
        discharge_add_ons = st.multiselect("Recommended Add-ons", ["Ezetimibe", "PCSK9 inhibitor", "Inclisiran"])
        target_ldl = st.slider("LDL-C Target (mmol/L)", 0.5, 3.0, 1.4, 0.1)
        patient_name = st.text_input("Patient Name for Report", placeholder="Enter patient name")

        submitted = st.form_submit_button("Calculate Treatment Impact")
        if submitted:
            projected_ldl, total_reduction = calculate_ldl_reduction(ldl, pre_statin, discharge_statin, discharge_add_ons)
            final_risk = calculate_ldl_effect(baseline_risk, ldl, projected_ldl, bp_controlled, not smoker, lifestyle)
            recommendations = generate_recommendations(final_risk)

            st.metric("Projected LDL-C", f"{projected_ldl:.1f} mmol/L", delta=f"{total_reduction:.0f}% reduction")
            st.metric("Post-Treatment Risk", f"{final_risk:.1f}%", delta=f"{baseline_risk - final_risk:.1f}% absolute reduction")

            st.subheader("Visual Risk Gauge")
            fig = plot_risk_gauge(final_risk)
            st.pyplot(fig)

            st.subheader("Clinical Recommendations")
            st.write(recommendations)
else:
    st.warning("Please complete all patient data to calculate risk.")

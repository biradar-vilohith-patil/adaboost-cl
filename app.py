import streamlit as st
from src.predict import run_inference

st.set_page_config(page_title="Career Placement AI", page_icon="🎓", layout="centered")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("🎓 Career Placement Predictor")
st.markdown("<p style='text-align: center; color: #94a3b8;'>Analyze your academic profile to determine your probability of corporate placement.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    ssc_p = st.number_input("10th Grade Percentage", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    hsc_p = st.number_input("12th Grade Percentage", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    degree_p = st.number_input("Undergrad Degree Percentage", min_value=0.0, max_value=100.0, value=65.0, step=1.0)

with col2:
    workex = st.selectbox("Prior Work Experience", ["Yes", "No"])
    specialisation = st.selectbox("MBA Specialization", ["Mkt&HR", "Mkt&Fin"])

if st.button("Evaluate Profile"):
    user_data = {
        'ssc_p': ssc_p,
        'hsc_p': hsc_p,
        'degree_p': degree_p,
        'workex': workex,
        'specialisation': specialisation
    }
    
    prediction, confidence = run_inference(user_data)
    
    if prediction == 1:
        st.success("Strong Candidate Profile")
        st.markdown(f"<h4 style='color: #4ade80; text-align: center;'>Placement Probability: {confidence:.1f}%</h4><p style='text-align: center; color: #f8fafc;'>Your current academic and experience metrics strongly align with successful corporate recruitment trends.</p>", unsafe_allow_html=True)
    else:
        st.warning("Profile Requires Enhancement")
        st.markdown(f"<h4 style='color: #fbbf24; text-align: center;'>Non-Placement Probability: {confidence:.1f}%</h4><p style='text-align: center; color: #f8fafc;'>Your profile indicates potential challenges in standard campus recruitment. Focus on gaining internship experience or improving specialized skill sets.</p>", unsafe_allow_html=True)
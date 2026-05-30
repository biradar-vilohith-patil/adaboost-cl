import streamlit as st
from src.predict import run_inference

st.set_page_config(page_title="Stock Screener AI", page_icon="📈", layout="centered")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("<div class='main-header'>📈 Stock Screener AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Identify long-term compounding fundamentals with institutional-grade AdaBoost classification.</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    roe = st.number_input("Return on Equity (ROE) %", min_value=-100.0, max_value=200.0, value=15.0, step=1.0)
    pe_ratio = st.number_input("P/E Ratio", min_value=0.0, max_value=500.0, value=25.0, step=1.0)
    debt_equity = st.number_input("Debt to Equity Ratio", min_value=0.0, max_value=10.0, value=0.5, step=0.1)

with col2:
    rev_growth = st.number_input("Revenue Growth (YoY) %", min_value=-100.0, max_value=500.0, value=12.0, step=1.0)
    op_margin = st.number_input("Operating Profit Margin %", min_value=-100.0, max_value=100.0, value=18.0, step=1.0)

if st.button("Initialize Fundamental Analysis"):
    # 1. Hard Guardrails (The Sanity Check)
    # Reject outright if Debt is toxic or ROE is terrible, regardless of the ML model
    if debt_equity > 1.5:
        st.markdown(f"""
        <div class='result-card error-card'>
            <div class='result-title'>AUTO-REJECT: TOXIC DEBT LEVEL</div>
            <div class='result-score'>D/E Ratio: {debt_equity}</div>
            <div class='result-desc'>This company is dangerously over-leveraged. High debt destroys long-term compounding. We do not need the AI to tell us this is a Value Trap.</div>
        </div>
        """, unsafe_allow_html=True)
        
    elif roe < 8.0:
        st.markdown(f"""
        <div class='result-card error-card'>
            <div class='result-title'>AUTO-REJECT: POOR CAPITAL EFFICIENCY</div>
            <div class='result-score'>ROE: {roe}%</div>
            <div class='result-desc'>A Return on Equity below 8% means the company cannot generate meaningful returns on shareholder capital. Skip.</div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # 2. If it passes the sanity check, let the AI evaluate the nuance
        user_data = {
            'returnOnEquity': roe / 100,
            'Debt to Equity': debt_equity,
            'PE ratio': pe_ratio,
            'Revenue Growth': rev_growth / 100,
            'operatingProfitMargin': op_margin / 100
        }
        
        prediction, confidence = run_inference(user_data)
        
        if prediction == 1:
            st.markdown(f"""
            <div class='result-card success-card'>
                <div class='result-title'>HIGH-CONVICTION MULTIBAGGER</div>
                <div class='result-score'>{confidence:.1f}% Confidence</div>
                <div class='result-desc'>Fundamentals demonstrate a strong economic moat. The balance sheet supports long-term wealth creation and sustainable compounding.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card error-card'>
                <div class='result-title'>VALUE TRAP / UNDERPERFORMER</div>
                <div class='result-score'>{confidence:.1f}% Confidence</div>
                <div class='result-desc'>Capital destruction risk detected by the model. Avoid deployment.</div>
            </div>
            """, unsafe_allow_html=True)
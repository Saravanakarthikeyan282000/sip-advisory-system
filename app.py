import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Recommendation", page_icon="📊", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #2c3e50; }
    div[data-testid="stExpander"] { background-color: white; border-radius: 10px; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG: EXCLUDED PAIRS ---
# (AMC, Scheme) tuples to strictly ignore
EXCLUDED_PAIRS = [
    ('Mirae', 'Smallcap'),
    ('Franklin', 'Multicap'),
    ('DSP', 'Multicap'),
    ('Kotak', 'Pharmafund'),
    ('HDFC', 'Pharmafund'),
    ('Mirae', 'Multicap'),
    ('Mirae', 'Flexicap'),
    ('UTI', 'Multicap')
]

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        rankings = pd.read_excel("25_Final_AHP_Ranking_5_1.xlsx")
        mc_results = pd.read_excel("26_Monte_Carlo_EWMA_Results.xlsx")
        forecasts = pd.read_excel("13_Forecasted_Fund_NAV.xlsx")
        
        # Clean column names
        rankings.columns = rankings.columns.str.strip()
        mc_results.columns = mc_results.columns.str.strip()
        forecasts.columns = forecasts.columns.str.strip()
        
        forecasts['Date'] = pd.to_datetime(forecasts['Date'])
        
        # --- FILTERING LOGIC ---
        # Remove excluded pairs from all datasets
        for amc, scheme in EXCLUDED_PAIRS:
            rankings = rankings[~((rankings['AMC'] == amc) & (rankings['Scheme'] == scheme))]
            mc_results = mc_results[~((mc_results['AMC'] == amc) & (mc_results['Scheme'] == scheme))]
            forecasts = forecasts[~((forecasts['AMC'] == amc) & (forecasts['Scheme'] == scheme))]
            
        return rankings, mc_results, forecasts
    except FileNotFoundError as e:
        st.error(f"⚠️ Missing File: {e}. Please ensure the Excel files are in the same folder as app.py")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_ranks, df_mc, df_forecast = load_data()

# --- HELPER FUNCTIONS ---
def format_currency(value):
    try:
        if pd.isna(value) or value == "": return "₹0"
        return f"₹{float(value):,.0f}"
    except: return "₹0"

def calculate_12m_forecast_sip(amc, scheme, monthly_sip):
    subset = df_forecast[(df_forecast['AMC'] == amc) & (df_forecast['Scheme'] == scheme)].sort_values('Date')
    if subset.empty: return 0
    subset['YearMonth'] = subset['Date'].dt.to_period('M')
    monthly_data = subset.groupby('YearMonth').first().head(12) 
    if len(monthly_data) < 12: return 0
    units = 0
    for nav in monthly_data['Forecast_NAV']:
        units += monthly_sip / nav
    final_nav = monthly_data.iloc[-1]['Forecast_NAV']
    return units * final_nav

def plot_bell_curve(p10, p50, p90, label, color='#4c78a8'):
    """Single Bell Curve Plot"""
    sigma = (p90 - p10) / 3.29 if p90 != p10 else p50 * 0.1
    x = np.linspace(p10 - sigma, p90 + sigma, 100)
    y = norm.pdf(x, p50, sigma)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=label, fill='tozeroy', line=dict(color=color, width=2)))
    
    # Add P10/P50/P90 Markers
    fig.add_trace(go.Scatter(x=[p10, p50, p90], y=[norm.pdf(p10,p50,sigma), norm.pdf(p50,p50,sigma), norm.pdf(p90,p50,sigma)],
                             mode='markers+text', text=['P10', 'P50', 'P90'], textposition="top center",
                             marker=dict(color=['red', 'gold', 'green'], size=8), showlegend=False))

    fig.update_layout(title=f"Probability Distribution: {label}", 
                      xaxis_title="Projected Value (₹)", yaxis_title="Probability",
                      height=350, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig

def plot_rebalance_comparison(curr_data, rec_data):
    """Comparison Bell Curve (Red vs Green)"""
    fig = go.Figure()

    # Current (Red)
    p50, p10, p90 = curr_data['P50'], curr_data['P10'], curr_data['P90']
    sigma = (p90 - p10) / 3.29 if p90 != p10 else p50 * 0.1
    x = np.linspace(p10 - sigma, p90 + sigma, 100)
    y = norm.pdf(x, p50, sigma)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"Current: {curr_data['AMC']}", 
                             fill='tozeroy', line=dict(color='#e74c3c', width=2), opacity=0.5))

    # Recommended (Green)
    p50r, p10r, p90r = rec_data['P50'], rec_data['P10'], rec_data['P90']
    sigmar = (p90r - p10r) / 3.29 if p90r != p10r else p50r * 0.1
    xr = np.linspace(p10r - sigmar, p90r + sigmar, 100)
    yr = norm.pdf(xr, p50r, sigmar)
    fig.add_trace(go.Scatter(x=xr, y=yr, mode='lines', name=f"Recommended: {rec_data['AMC']}", 
                             fill='tozeroy', line=dict(color='#2ecc71', width=2), opacity=0.6))

    fig.update_layout(title=f"Rebalancing Impact: {curr_data['AMC']} vs {rec_data['AMC']}", 
                      xaxis_title="Projected Value (₹)", yaxis_title="Probability",
                      height=400, margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(orientation="h", y=1.1))
    return fig

# --- NAVIGATION ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["First-Time Investor", "Existing Investor"])

# ==========================================
# PAGE: FIRST-TIME INVESTOR
# ==========================================
if page == "First-Time Investor":
    st.title("🎯 New Investment Recommendation")
    
    if df_mc.empty: st.stop()

    col1, col2 = st.columns(2)
    with col1:
        schemes = sorted(df_mc['Scheme'].unique())
        selected_scheme = st.selectbox("Select Scheme Category:", schemes)
        amounts = sorted(df_mc['SIP_Amount'].unique())
        sip_amount = st.selectbox("Monthly SIP Amount (₹):", amounts, index=amounts.index(5000) if 5000 in amounts else 0)

    with col2:
        duration = st.selectbox("Duration (Months):", [12, 24, 36])
        top_n = st.selectbox("AMCs to be recommended:", [1, 2, 3, 4, 5], index=0)

    if st.button("Generate Recommendation"):
        st.divider()
        
        # Rank Logic
        ranked_subset = df_ranks[df_ranks['Scheme'] == selected_scheme].sort_values('Final_Score', ascending=False).head(top_n)
        top_amcs = ranked_subset['AMC'].tolist()
        
        if not top_amcs:
            st.warning("No valid recommendations found after filtering.")
            st.stop()

        # Iterate through EACH recommended AMC
        for i, amc in enumerate(top_amcs):
            invested = sip_amount * duration
            
            # Calculate P50/P10/P90
            if duration == 12:
                expected_val = calculate_12m_forecast_sip(amc, selected_scheme, sip_amount)
                p50, p10, p90 = expected_val, expected_val * 0.95, expected_val * 1.05
            else:
                row = df_mc[(df_mc['AMC'] == amc) & (df_mc['Scheme'] == selected_scheme) & 
                            (df_mc['SIP_Amount'] == sip_amount) & (df_mc['Tenure_Months'] == duration)]
                if not row.empty:
                    p50, p10, p90 = row.iloc[0]['P50_Corpus'], row.iloc[0]['P10_Corpus'], row.iloc[0]['P90_Corpus']
                else:
                    p50, p10, p90 = 0, 0, 0

            if p50 > 0:
                # --- DISPLAY CARD FOR EACH AMC ---
                with st.container():
                    st.markdown(f"### #{i+1} Recommendation: {amc}")
                    
                    # Metrics Row
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Investment", format_currency(invested))
                    m2.metric("Expected (P50)", format_currency(p50))
                    m3.metric("Optimistic (P90)", format_currency(p90))
                    m4.metric("Pessimistic (P10)", format_currency(p10))
                    
                    # Bell Curve Row
                    st.plotly_chart(plot_bell_curve(p10, p50, p90, amc), use_container_width=True)
                    st.divider()

# ==========================================
# PAGE: EXISTING INVESTOR
# ==========================================
elif page == "Existing Investor":
    st.title("🔄 Portfolio Rebalancing")
    
    if df_mc.empty: st.stop()

    num_funds = st.number_input("Current Portfolio Count:", min_value=1, max_value=10, step=1, value=1)
    
    user_portfolio = []
    st.markdown("### Enter Portfolio Details")
    
    with st.form("portfolio_form"):
        for i in range(num_funds):
            with st.expander(f"Fund {i+1} Input", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                # Dropdowns filtered automatically by load_data
                sch = c1.selectbox(f"Scheme", sorted(df_mc['Scheme'].unique()), key=f"s_{i}")
                
                # Filter AMCs available for this scheme
                avail_amcs = sorted(df_mc[df_mc['Scheme'] == sch]['AMC'].unique())
                amc = c2.selectbox(f"AMC", avail_amcs, key=f"a_{i}")
                
                amt = c3.selectbox(f"SIP Amount", sorted(df_mc['SIP_Amount'].unique()), index=9, key=f"m_{i}")
                ten = c4.selectbox(f"Tenure", [12, 24, 36], index=1, key=f"t_{i}")
                user_portfolio.append({'Scheme': sch, 'AMC': amc, 'Amount': amt, 'Tenure': ten})
        
        submitted = st.form_submit_button("Analyze & Rebalance")

    if submitted:
        st.divider()
        st.markdown("## 📊 Fund-wise Rebalancing Report")
        
        for idx, fund in enumerate(user_portfolio):
            
            # --- 1. DATA RETRIEVAL ---
            # Current Fund
            if fund['Tenure'] == 12:
                curr_p50 = calculate_12m_forecast_sip(fund['AMC'], fund['Scheme'], fund['Amount'])
                curr_p10, curr_p90 = curr_p50 * 0.95, curr_p50 * 1.05
            else:
                curr_row = df_mc[(df_mc['AMC'] == fund['AMC']) & (df_mc['Scheme'] == fund['Scheme']) & 
                                 (df_mc['SIP_Amount'] == fund['Amount']) & (df_mc['Tenure_Months'] == fund['Tenure'])]
                curr_p50 = curr_row.iloc[0]['P50_Corpus'] if not curr_row.empty else 0
                curr_p10 = curr_row.iloc[0]['P10_Corpus'] if not curr_row.empty else 0
                curr_p90 = curr_row.iloc[0]['P90_Corpus'] if not curr_row.empty else 0

            # Recommended Fund
            best_amc_name = df_ranks[df_ranks['Scheme'] == fund['Scheme']].sort_values('Final_Score', ascending=False).iloc[0]['AMC']
            
            if fund['Tenure'] == 12:
                best_p50 = calculate_12m_forecast_sip(best_amc_name, fund['Scheme'], fund['Amount'])
                best_p10, best_p90 = best_p50 * 0.95, best_p50 * 1.05
            else:
                best_row = df_mc[(df_mc['AMC'] == best_amc_name) & (df_mc['Scheme'] == fund['Scheme']) & 
                                 (df_mc['SIP_Amount'] == fund['Amount']) & (df_mc['Tenure_Months'] == fund['Tenure'])]
                best_p50 = best_row.iloc[0]['P50_Corpus'] if not best_row.empty else 0
                best_p10 = best_row.iloc[0]['P10_Corpus'] if not best_row.empty else 0
                best_p90 = best_row.iloc[0]['P90_Corpus'] if not best_row.empty else 0

            # --- 2. DISPLAY LOGIC ---
            if curr_p50 > 0 and best_p50 > 0:
                gain = best_p50 - curr_p50
                is_switch = (gain > 500) and (fund['AMC'] != best_amc_name)
                
                # Layout
                st.markdown(f"### Fund {idx+1}: {fund['Scheme']}")
                col_kpi, col_chart = st.columns([1, 2])
                
                with col_kpi:
                    st.write(f"Current: **{fund['AMC']}**")
                    st.metric("Current Projected Value", format_currency(curr_p50))
                    
                    if is_switch:
                        st.markdown("---")
                        st.error(f"⚠️ Recommendation: **SWITCH**")
                        st.write(f"To: **{best_amc_name}**")
                        st.metric("Potential Gain", format_currency(gain), delta="Profit Opportunity")
                    else:
                        st.markdown("---")
                        st.success("✅ Recommendation: **HOLD**")
                        st.write("You are already in the best fund.")
                
                with col_chart:
                    curr_data = {'AMC': fund['AMC'], 'P50': curr_p50, 'P10': curr_p10, 'P90': curr_p90}
                    rec_data = {'AMC': best_amc_name, 'P50': best_p50, 'P10': best_p10, 'P90': best_p90}
                    st.plotly_chart(plot_rebalance_comparison(curr_data, rec_data), use_container_width=True)
                
                st.divider()
            else:
                st.warning(f"Insufficient data for {fund['AMC']} - {fund['Scheme']}")
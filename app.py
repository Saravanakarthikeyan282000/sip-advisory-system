import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimization System", layout="wide")

# --- PROFESSIONAL STYLING (DARK MODE + CYAN) ---
st.markdown("""
    <style>
    body { color: #e0e0e0; background-color: #0e1117; }
    h1, h2, h3, h4 { color: #00FFFF !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    div[data-testid="stMetricValue"] { color: #00FFFF !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #b0b0b0 !important; }
    .stMetric { background-color: #262730; border: 1px solid #464b5c; border-radius: 5px; padding: 10px; }
    .stSelectbox label, .stNumberInput label { color: #00FFFF !important; font-weight: bold; }
    div[data-testid="stDataFrame"] { border: 1px solid #464b5c; }
    hr { border-color: #00FFFF; margin-top: 2rem; margin-bottom: 2rem; opacity: 0.3; }
    .footer {
        position: fixed;
        left: 0; bottom: 0; width: 100%;
        background-color: #0e1117; color: #00FFFF;
        text-align: center; padding: 10px; font-weight: bold;
        border-top: 1px solid #464b5c; z-index: 100;
    }
    .streamlit-expanderHeader { color: #ffffff; font-weight: bold; background-color: #262730; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG: EXCLUDED PAIRS ---
EXCLUDED_PAIRS = [
    ('Mirae', 'Smallcap'), ('Franklin', 'Multicap'), ('DSP', 'Multicap'),
    ('Kotak', 'Pharmafund'), ('HDFC', 'Pharmafund'), ('Mirae', 'Multicap'),
    ('Mirae', 'Flexicap'), ('UTI', 'Multicap')
]

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        rankings = pd.read_excel("25_Final_AHP_Ranking_5_1.xlsx")
        mc_results = pd.read_excel("26_Monte_Carlo_EWMA_Results.xlsx")
        forecasts = pd.read_excel("13_Forecasted_Fund_NAV.xlsx")
        
        # Standardize Strings
        rankings.columns = rankings.columns.str.strip()
        mc_results.columns = mc_results.columns.str.strip()
        forecasts.columns = forecasts.columns.str.strip()
        
        # --- CRITICAL: FORCE INTEGER TYPES FOR ACCURATE LOOKUP ---
        # This prevents "5000" (text) vs 5000 (number) mismatches
        mc_results['SIP_Amount'] = pd.to_numeric(mc_results['SIP_Amount'], errors='coerce').fillna(0).astype(int)
        mc_results['Tenure_Months'] = pd.to_numeric(mc_results['Tenure_Months'], errors='coerce').fillna(0).astype(int)
        
        forecasts['Date'] = pd.to_datetime(forecasts['Date'])
        
        # Remove Exclusions
        for amc, scheme in EXCLUDED_PAIRS:
            rankings = rankings[~((rankings['AMC'] == amc) & (rankings['Scheme'] == scheme))]
            mc_results = mc_results[~((mc_results['AMC'] == amc) & (mc_results['Scheme'] == scheme))]
            forecasts = forecasts[~((forecasts['AMC'] == amc) & (forecasts['Scheme'] == scheme))]
            
        return rankings, mc_results, forecasts
    except FileNotFoundError as e:
        st.error(f"System Error: Required data file not found ({e}).")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_ranks, df_mc, df_forecast = load_data()

# --- UTILITY FUNCTIONS ---
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
    return units * monthly_data.iloc[-1]['Forecast_NAV']

def generate_bell_curve(curr_data, title_text="Probability Distribution", color_code='#00FFFF'):
    fig = go.Figure()
    p50, p10, p90 = curr_data['P50'], curr_data['P10'], curr_data['P90']
    sigma = (p90 - p10) / 3.29 if p90 != p10 else p50 * 0.1
    x = np.linspace(p10 - sigma, p90 + sigma, 100)
    y = norm.pdf(x, p50, sigma)
    
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='tozeroy', line=dict(color=color_code, width=2), opacity=0.4))
    fig.add_trace(go.Scatter(x=[p10], y=[norm.pdf(p10,p50,sigma)], mode='markers+text', text=['P10'], textposition="bottom center", marker=dict(color='#FF4B4B', size=10)))
    fig.add_trace(go.Scatter(x=[p50], y=[norm.pdf(p50,p50,sigma)], mode='markers+text', text=['P50'], textposition="top center", marker=dict(color='#FFFFFF', size=10)))
    fig.add_trace(go.Scatter(x=[p90], y=[norm.pdf(p90,p50,sigma)], mode='markers+text', text=['P90'], textposition="bottom center", marker=dict(color='#00FF00', size=10)))
    
    fig.update_layout(title=dict(text=title_text, font=dict(size=14, color='#b0b0b0')), template="plotly_dark", 
                      xaxis_title="Corpus Value (₹)", yaxis=dict(showticklabels=False), height=300, showlegend=False, 
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- NAVIGATION ---
st.sidebar.title("System Navigation")
page = st.sidebar.radio("Select Module:", ["New Investment Analysis", "Existing Portfolio Rebalancing"])

# =========================================================
# MODULE 1: NEW INVESTMENT
# =========================================================
if page == "New Investment Analysis":
    st.title("New Investment Analysis")
    st.markdown("Generate optimized portfolio recommendations.")
    if df_mc.empty: st.stop()

    st.subheader("Investment Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        schemes = sorted(df_mc['Scheme'].unique())
        sel_scheme = st.selectbox("Scheme Category", schemes)
    with c2:
        amounts = sorted(df_mc['SIP_Amount'].unique())
        sel_sip = st.selectbox("Monthly SIP Amount", amounts, index=amounts.index(5000) if 5000 in amounts else 0)
    with c3:
        sel_duration = st.selectbox("Duration (Months)", [12, 24, 36])
    with c4:
        sel_top_n = st.selectbox("Recommendations Count", [1, 2, 3, 4, 5], index=0)

    if st.button("RUN ANALYSIS", type="primary"):
        st.divider()
        ranked_subset = df_ranks[df_ranks['Scheme'] == sel_scheme].sort_values('Final_Score', ascending=False).head(sel_top_n)
        top_amcs = ranked_subset['AMC'].tolist()
        
        for i, amc in enumerate(top_amcs):
            invested = sel_sip * sel_duration
            if sel_duration == 12:
                expected_val = calculate_12m_forecast_sip(amc, sel_scheme, sel_sip)
                p50, p10, p90 = expected_val, expected_val * 0.95, expected_val * 1.05
            else:
                row = df_mc[(df_mc['AMC'] == amc) & (df_mc['Scheme'] == sel_scheme) & 
                            (df_mc['SIP_Amount'] == sel_sip) & (df_mc['Tenure_Months'] == sel_duration)]
                if not row.empty:
                    p50, p10, p90 = row.iloc[0]['P50_Corpus'], row.iloc[0]['P10_Corpus'], row.iloc[0]['P90_Corpus']
                else: p50, p10, p90 = 0, 0, 0

            if p50 > 0:
                with st.container():
                    st.markdown(f"### Rank {i+1}: {amc} {sel_scheme}")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Invested Capital", format_currency(invested))
                    m2.metric("Expected (P50)", format_currency(p50))
                    m3.metric("Optimistic (P90)", format_currency(p90))
                    m4.metric("Pessimistic (P10)", format_currency(p10))
                    curr_data = {'AMC': amc, 'P50': p50, 'P10': p10, 'P90': p90}
                    st.plotly_chart(generate_bell_curve(curr_data, title_text=f"Projected Outcome: {amc}", color_code='#00FFFF'), use_container_width=True)
                    st.markdown("---")

# =========================================================
# MODULE 2: EXISTING PORTFOLIO REBALANCING
# =========================================================
elif page == "Existing Portfolio Rebalancing":
    st.title("Existing Portfolio Rebalancing")
    st.markdown("Comparative analysis: Checks if a better performing fund exists for your exact parameters.")
    if df_mc.empty: st.stop()

    num_funds = st.number_input("Number of Holdings", min_value=1, max_value=20, step=1, value=1)
    user_portfolio = []
    
    st.subheader("Portfolio Composition")
    for i in range(num_funds):
        with st.expander(f"Holding #{i+1}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            sch = c1.selectbox("Scheme", sorted(df_mc['Scheme'].unique()), key=f"s_{i}")
            
            # Dynamic Filter: Only valid AMCs
            valid_amcs = sorted(df_mc[df_mc['Scheme'] == sch]['AMC'].unique())
            if not valid_amcs:
                amc = None
                c2.error("No AMCs found.")
            else:
                amc = c2.selectbox("AMC", valid_amcs, key=f"a_{i}")
            
            amt = c3.selectbox("SIP Amount", sorted(df_mc['SIP_Amount'].unique()), index=9, key=f"m_{i}")
            ten = c4.selectbox("Tenure", [12, 24, 36], index=1, key=f"t_{i}")
            
            if amc: user_portfolio.append({'id': i+1, 'Scheme': sch, 'AMC': amc, 'Amount': amt, 'Tenure': ten})
    
    if st.button("ANALYZE & REBALANCE", type="primary"):
        st.divider()
        st.subheader("Rebalancing Analysis Report")
        
        summary_table_data = [] 
        total_current_val = 0
        total_optimized_val = 0
        
        for fund in user_portfolio:
            # 1. FETCH CURRENT FUND PERFORMANCE (Use exact Amount/Tenure inputs)
            if fund['Tenure'] == 12:
                c_p50 = calculate_12m_forecast_sip(fund['AMC'], fund['Scheme'], fund['Amount'])
                c_p10, c_p90 = c_p50 * 0.95, c_p50 * 1.05
            else:
                row = df_mc[
                    (df_mc['AMC'] == fund['AMC']) & 
                    (df_mc['Scheme'] == fund['Scheme']) & 
                    (df_mc['SIP_Amount'] == int(fund['Amount'])) &    # Strict Integer Match
                    (df_mc['Tenure_Months'] == int(fund['Tenure']))   # Strict Integer Match
                ]
                c_p50 = row.iloc[0]['P50_Corpus'] if not row.empty else 0
                c_p10 = row.iloc[0]['P10_Corpus'] if not row.empty else 0
                c_p90 = row.iloc[0]['P90_Corpus'] if not row.empty else 0

            # 2. FIND THE WINNER (For this specific Scheme + Amount + Tenure)
            best_amc = fund['AMC']
            b_p50, b_p10, b_p90 = c_p50, c_p10, c_p90
            
            if fund['Tenure'] == 12:
                all_candidates = df_forecast[df_forecast['Scheme'] == fund['Scheme']]['AMC'].unique()
                max_val = c_p50
                for cand in all_candidates:
                    val = calculate_12m_forecast_sip(cand, fund['Scheme'], fund['Amount'])
                    if val > max_val:
                        max_val = val
                        best_amc = cand
                        b_p50 = val
                        b_p10, b_p90 = val * 0.95, val * 1.05
            else:
                # Filter MC data for ALL AMCs matching this Scheme + Amount + Tenure
                # This ensures if you change Amount, you get the winner for the NEW amount.
                cohort = df_mc[
                    (df_mc['Scheme'] == fund['Scheme']) & 
                    (df_mc['SIP_Amount'] == int(fund['Amount'])) & 
                    (df_mc['Tenure_Months'] == int(fund['Tenure']))
                ]
                
                if not cohort.empty:
                    # Find the AMC with the MAX P50 in this specific cohort
                    best_row = cohort.loc[cohort['P50_Corpus'].idxmax()]
                    if best_row['P50_Corpus'] > c_p50:
                        best_amc = best_row['AMC']
                        b_p50 = best_row['P50_Corpus']
                        b_p10 = best_row['P10_Corpus']
                        b_p90 = best_row['P90_Corpus']

            # 3. DECISION LOGIC
            gain = b_p50 - c_p50
            
            if gain > 50 and best_amc != fund['AMC']: 
                action = "REBALANCE"
                action_color = "#FF4B4B" 
                display_best_amc = best_amc
                display_gain = format_currency(gain)
            else:
                action = "HOLD"
                action_color = "#00FFFF"
                gain = 0
                display_best_amc = "Not Required"
                display_gain = "-"
                b_p50 = c_p50 

            total_current_val += c_p50
            total_optimized_val += b_p50

            # 4. DISPLAY
            st.markdown(f"#### Holding #{fund['id']}: {fund['Scheme']} - {fund['AMC']}")
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Recommendation", action)
            c_m2.metric("Best Alternative", display_best_amc)
            c_m3.metric("Projected Gain", display_gain)

            if action == "REBALANCE":
                g1, g2 = st.columns(2)
                with g1:
                    st.plotly_chart(generate_bell_curve({'AMC':fund['AMC'],'P50':c_p50,'P10':c_p10,'P90':c_p90}, title_text=f"Current: {fund['AMC']}", color_code='#FF4B4B'), use_container_width=True)
                with g2:
                    st.plotly_chart(generate_bell_curve({'AMC':best_amc,'P50':b_p50,'P10':b_p10,'P90':b_p90}, title_text=f"Proposed: {best_amc}", color_code='#00FF00'), use_container_width=True)
            else:
                st.plotly_chart(generate_bell_curve({'AMC':fund['AMC'],'P50':c_p50,'P10':c_p10,'P90':c_p90}, title_text=f"Current Performance: {fund['AMC']}", color_code='#00FFFF'), use_container_width=True)
                st.caption(f"Correct Choice! {fund['AMC']} is the top performer in this category.")
            st.markdown("---")

            summary_table_data.append({
                "Holding": f"#{fund['id']}", "Scheme": fund['Scheme'], "Current AMC": fund['AMC'],
                "Action": action, "Switch To": display_best_amc, "Gain": display_gain
            })

        # 5. SUMMARY
        st.subheader("Consolidated Portfolio Report")
        if summary_table_data:
            summary_df = pd.DataFrame(summary_table_data)
            st.dataframe(summary_df, use_container_width=True)

        st.subheader("Wealth Impact Analysis")
        bar_data = pd.DataFrame({
            "State": ["Current Portfolio", "Optimized Portfolio"],
            "Value": [total_current_val, total_optimized_val],
            "Color": ["#00FFFF", "#2ecc71"]
        })
        fig_bar = px.bar(bar_data, x="State", y="Value", text="Value", color="State", color_discrete_sequence=["#00FFFF", "#2ecc71"])
        fig_bar.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        fig_bar.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        csv = convert_df_to_csv(summary_df)
        st.download_button(label="DOWNLOAD REPORT (CSV)", data=csv, file_name='portfolio_rebalancing_report.csv', mime='text/csv', type="primary")

# --- FOOTER ---
st.markdown("""<div class="footer">Designed by Mr.Saravana Karthikeyan K</div>""", unsafe_allow_html=True)
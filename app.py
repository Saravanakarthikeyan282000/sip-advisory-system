import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimization System", layout="wide")

# --- PROFESSIONAL STYLING (DARK MODE + CYAN) ---
st.markdown("""
    <style>
    /* Global Text Color */
    body { color: #e0e0e0; background-color: #0e1117; }
    
    /* Headers - Cyan Accent */
    h1, h2, h3, h4 { color: #00FFFF !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    
    /* Metric Boxes */
    div[data-testid="stMetricValue"] { color: #00FFFF !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #b0b0b0 !important; }
    .stMetric { background-color: #262730; border: 1px solid #464b5c; border-radius: 5px; padding: 10px; }
    
    /* Input Fields */
    .stSelectbox label, .stNumberInput label { color: #00FFFF !important; font-weight: bold; }
    
    /* Dataframes */
    div[data-testid="stDataFrame"] { border: 1px solid #464b5c; }
    
    /* Divider */
    hr { border-color: #00FFFF; margin-top: 2rem; margin-bottom: 2rem; opacity: 0.3; }
    
    /* Expander Styling */
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
        
        # Standardization
        rankings.columns = rankings.columns.str.strip()
        mc_results.columns = mc_results.columns.str.strip()
        forecasts.columns = forecasts.columns.str.strip()
        forecasts['Date'] = pd.to_datetime(forecasts['Date'])
        
        # Filtering Exclusions
        for amc, scheme in EXCLUDED_PAIRS:
            rankings = rankings[~((rankings['AMC'] == amc) & (rankings['Scheme'] == scheme))]
            mc_results = mc_results[~((mc_results['AMC'] == amc) & (mc_results['Scheme'] == scheme))]
            forecasts = forecasts[~((forecasts['AMC'] == amc) & (forecasts['Scheme'] == scheme))]
            
        return rankings, mc_results, forecasts
    except FileNotFoundError as e:
        st.error(f"System Error: Required data file not found ({e}). Contact administrator.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_ranks, df_mc, df_forecast = load_data()

# --- UTILITY FUNCTIONS ---
def format_currency(value):
    try:
        if pd.isna(value) or value == "": return "0"
        return f"{float(value):,.0f}"
    except: return "0"

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

def generate_bell_curve(curr_data, rec_data=None):
    """
    Generates a professional comparison chart.
    Cyan = Recommended / Best
    Red = Current (if inferior)
    """
    fig = go.Figure()

    # Plot Current Fund (Red)
    p50, p10, p90 = curr_data['P50'], curr_data['P10'], curr_data['P90']
    sigma = (p90 - p10) / 3.29 if p90 != p10 else p50 * 0.1
    x = np.linspace(p10 - sigma, p90 + sigma, 100)
    y = norm.pdf(x, p50, sigma)
    
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', name=f"Current: {curr_data['AMC']}", 
        fill='tozeroy', line=dict(color='#FF4B4B', width=2), opacity=0.4
    ))

    # Plot Recommended Fund (Cyan) if distinct
    if rec_data and rec_data['AMC'] != curr_data['AMC']:
        p50r, p10r, p90r = rec_data['P50'], rec_data['P10'], rec_data['P90']
        sigmar = (p90r - p10r) / 3.29 if p90r != p10r else p50r * 0.1
        xr = np.linspace(p10r - sigmar, p90r + sigmar, 100)
        yr = norm.pdf(xr, p50r, sigmar)
        
        fig.add_trace(go.Scatter(
            x=xr, y=yr, mode='lines', name=f"Recommended: {rec_data['AMC']}", 
            fill='tozeroy', line=dict(color='#00FFFF', width=3), opacity=0.6
        ))

    # Dark Mode Layout
    fig.update_layout(
        title="Probabilistic Outcome Distribution",
        template="plotly_dark",
        xaxis_title="Projected Corpus Value",
        yaxis_title="Probability Density",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- MAIN NAVIGATION ---
st.sidebar.title("System Navigation")
page = st.sidebar.radio("Select Module:", ["New Investment Analysis", "Existing Portfolio Rebalancing"])

# =========================================================
# MODULE 1: NEW INVESTMENT ANALYSIS
# =========================================================
if page == "New Investment Analysis":
    st.title("New Investment Analysis")
    st.markdown("Generate optimized portfolio recommendations based on Monte Carlo simulations and CNN-GRU forecasts.")
    
    if df_mc.empty: st.stop()

    # INPUT SECTION
    with st.container():
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
        
        # LOGIC: Filter by Rank
        ranked_subset = df_ranks[df_ranks['Scheme'] == sel_scheme].sort_values('Final_Score', ascending=False).head(sel_top_n)
        top_amcs = ranked_subset['AMC'].tolist()
        
        if not top_amcs:
            st.error("No valid funds found matching criteria.")
            st.stop()

        for i, amc in enumerate(top_amcs):
            invested = sel_sip * sel_duration
            
            # Data Retrieval
            if sel_duration == 12:
                expected_val = calculate_12m_forecast_sip(amc, sel_scheme, sel_sip)
                p50, p10, p90 = expected_val, expected_val * 0.95, expected_val * 1.05
            else:
                row = df_mc[(df_mc['AMC'] == amc) & (df_mc['Scheme'] == sel_scheme) & 
                            (df_mc['SIP_Amount'] == sel_sip) & (df_mc['Tenure_Months'] == sel_duration)]
                if not row.empty:
                    p50, p10, p90 = row.iloc[0]['P50_Corpus'], row.iloc[0]['P10_Corpus'], row.iloc[0]['P90_Corpus']
                else:
                    p50, p10, p90 = 0, 0, 0

            if p50 > 0:
                # CARD VIEW
                with st.container():
                    st.markdown(f"### Rank {i+1}: {amc} {sel_scheme}")
                    
                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Invested Capital", format_currency(invested))
                    m2.metric("Projected Value (P50)", format_currency(p50))
                    m3.metric("Optimistic (P90)", format_currency(p90))
                    m4.metric("Conservative (P10)", format_currency(p10))
                    
                    # Chart
                    curr_data = {'AMC': amc, 'P50': p50, 'P10': p10, 'P90': p90}
                    # For new investors, we only show the recommended curve (pass same data twice or handle in func)
                    # We will reuse the comparison function but pass identical data to show single curve styled as 'Recommended'
                    st.plotly_chart(generate_bell_curve(curr_data, curr_data), use_container_width=True)
                    st.markdown("---")

# =========================================================
# MODULE 2: EXISTING PORTFOLIO REBALANCING
# =========================================================
elif page == "Existing Portfolio Rebalancing":
    st.title("Existing Portfolio Rebalancing")
    st.markdown("Comparative analysis of current holdings against top-ranked market alternatives.")
    
    if df_mc.empty: st.stop()

    num_funds = st.number_input("Number of Holdings", min_value=1, max_value=20, step=1, value=1)
    
    user_portfolio = []
    
    with st.form("rebalance_form"):
        st.subheader("Portfolio Composition")
        for i in range(num_funds):
            with st.expander(f"Holding #{i+1}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                sch = c1.selectbox("Scheme", sorted(df_mc['Scheme'].unique()), key=f"s_{i}")
                
                # Dynamic AMC Filter
                avail_amcs = sorted(df_mc[df_mc['Scheme'] == sch]['AMC'].unique())
                amc = c2.selectbox("AMC", avail_amcs, key=f"a_{i}")
                
                amt = c3.selectbox("SIP Amount", sorted(df_mc['SIP_Amount'].unique()), index=9, key=f"m_{i}")
                ten = c4.selectbox("Tenure", [12, 24, 36], index=1, key=f"t_{i}")
                
                user_portfolio.append({'id': i+1, 'Scheme': sch, 'AMC': amc, 'Amount': amt, 'Tenure': ten})
        
        submitted = st.form_submit_button("ANALYZE & REBALANCE", type="primary")

    if submitted:
        st.divider()
        st.subheader("Rebalancing Analysis Report")
        
        report_data = [] # For CSV download
        total_invested = 0
        total_current_val = 0
        total_optimized_val = 0

        for fund in user_portfolio:
            # 1. FETCH CURRENT FUND DATA
            if fund['Tenure'] == 12:
                c_p50 = calculate_12m_forecast_sip(fund['AMC'], fund['Scheme'], fund['Amount'])
                c_p10, c_p90 = c_p50 * 0.95, c_p50 * 1.05
            else:
                row = df_mc[(df_mc['AMC'] == fund['AMC']) & (df_mc['Scheme'] == fund['Scheme']) & 
                            (df_mc['SIP_Amount'] == fund['Amount']) & (df_mc['Tenure_Months'] == fund['Tenure'])]
                c_p50 = row.iloc[0]['P50_Corpus'] if not row.empty else 0
                c_p10 = row.iloc[0]['P10_Corpus'] if not row.empty else 0
                c_p90 = row.iloc[0]['P90_Corpus'] if not row.empty else 0

            # 2. IDENTIFY TOP RANKED ALTERNATIVE
            try:
                # Sort rankings to find #1 for this scheme
                top_rank_row = df_ranks[df_ranks['Scheme'] == fund['Scheme']].sort_values('Final_Score', ascending=False).iloc[0]
                best_amc = top_rank_row['AMC']
                
                if fund['Tenure'] == 12:
                    b_p50 = calculate_12m_forecast_sip(best_amc, fund['Scheme'], fund['Amount'])
                    b_p10, b_p90 = b_p50 * 0.95, b_p50 * 1.05
                else:
                    brow = df_mc[(df_mc['AMC'] == best_amc) & (df_mc['Scheme'] == fund['Scheme']) & 
                                 (df_mc['SIP_Amount'] == fund['Amount']) & (df_mc['Tenure_Months'] == fund['Tenure'])]
                    b_p50 = brow.iloc[0]['P50_Corpus'] if not brow.empty else 0
                    b_p10 = brow.iloc[0]['P10_Corpus'] if not brow.empty else 0
                    b_p90 = brow.iloc[0]['P90_Corpus'] if not brow.empty else 0
                
            except IndexError:
                best_amc = fund['AMC']
                b_p50, b_p10, b_p90 = c_p50, c_p10, c_p90

            # 3. ANALYSIS LOGIC
            invested = fund['Amount'] * fund['Tenure']
            gain = b_p50 - c_p50
            action = "HOLD"
            
            if fund['AMC'] == best_amc:
                action = "HOLD (Top Ranked)"
            elif gain > 0:
                action = "REBALANCE"
            else:
                action = "HOLD (No Better Alternative)"

            # Accumulate Totals
            total_invested += invested
            total_current_val += c_p50
            total_optimized_val += b_p50
            
            # Store Data for CSV
            report_data.append({
                "Holding ID": fund['id'],
                "Scheme": fund['Scheme'],
                "Current AMC": fund['AMC'],
                "Recommended AMC": best_amc,
                "Action": action,
                "Invested": invested,
                "Current Projected": c_p50,
                "Optimized Projected": b_p50,
                "Potential Gain": gain
            })

            # 4. DISPLAY INDIVIDUAL ANALYSIS
            st.markdown(f"#### Holding #{fund['id']}: {fund['Scheme']}")
            col_metrics, col_graph = st.columns([1, 2])
            
            with col_metrics:
                st.write(f"Current: **{fund['AMC']}**")
                st.write(f"Top Ranked: **{best_amc}**")
                
                st.metric("Action", action)
                if action == "REBALANCE":
                    st.metric("Potential Gain", format_currency(gain), delta="Optimization Benefit")
                
            with col_graph:
                curr_dict = {'AMC': fund['AMC'], 'P50': c_p50, 'P10': c_p10, 'P90': c_p90}
                rec_dict = {'AMC': best_amc, 'P50': b_p50, 'P10': b_p10, 'P90': b_p90}
                st.plotly_chart(generate_bell_curve(curr_dict, rec_dict), use_container_width=True)
            
            st.markdown("---")

        # 5. FINAL SUMMARY SECTION
        st.subheader("Consolidated Portfolio Summary")
        
        sum_c1, sum_c2, sum_c3 = st.columns(3)
        sum_c1.metric("Total Invested Capital", format_currency(total_invested))
        sum_c2.metric("Current Portfolio Projection", format_currency(total_current_val))
        
        net_gain = total_optimized_val - total_current_val
        sum_c3.metric("Optimized Portfolio Projection", format_currency(total_optimized_val), delta=format_currency(net_gain))

        # 6. DOWNLOAD REPORT
        st.subheader("Export Analysis")
        csv_df = pd.DataFrame(report_data)
        csv = convert_df_to_csv(csv_df)
        
        st.download_button(
            label="DOWNLOAD FULL REPORT (CSV)",
            data=csv,
            file_name='portfolio_rebalancing_report.csv',
            mime='text/csv',
            type="primary"
        )
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

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load files from the main folder
        rankings = pd.read_excel("25_Final_AHP_Ranking_5_1.xlsx")
        mc_results = pd.read_excel("26_Monte_Carlo_EWMA_Results.xlsx")
        forecasts = pd.read_excel("13_Forecasted_Fund_NAV.xlsx")
        
        # Clean column names
        rankings.columns = rankings.columns.str.strip()
        mc_results.columns = mc_results.columns.str.strip()
        forecasts.columns = forecasts.columns.str.strip()
        
        # Ensure Date is datetime
        forecasts['Date'] = pd.to_datetime(forecasts['Date'])
        
        return rankings, mc_results, forecasts
    except FileNotFoundError as e:
        st.error(f"⚠️ Missing File: {e}. Please ensure the Excel files are in the same folder as app.py")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_ranks, df_mc, df_forecast = load_data()

# --- HELPER FUNCTIONS ---
def format_currency(value):
    """Safely formats currency, handling NaNs or strings"""
    try:
        if pd.isna(value) or value == "":
            return "₹0"
        return f"₹{float(value):,.0f}"
    except:
        return "₹0"

def calculate_12m_forecast_sip(amc, scheme, monthly_sip):
    """Calculates SIP returns using the raw 12-month Forecasted NAV data"""
    subset = df_forecast[(df_forecast['AMC'] == amc) & (df_forecast['Scheme'] == scheme)].sort_values('Date')
    if subset.empty: return 0
    
    # Simulate SIP: Buy units on the 1st of every month for 12 months
    subset['YearMonth'] = subset['Date'].dt.to_period('M')
    monthly_data = subset.groupby('YearMonth').first().head(12) 
    
    if len(monthly_data) < 12: return 0
    
    units = 0
    for nav in monthly_data['Forecast_NAV']:
        units += monthly_sip / nav
        
    final_nav = monthly_data.iloc[-1]['Forecast_NAV']
    return units * final_nav

def plot_comparison_bell_curve(curr_data, rec_data=None):
    """
    Plots Bell Curves. 
    If rec_data is provided, plots both Current (Red) and Recommended (Green) for comparison.
    """
    fig = go.Figure()

    # 1. Plot Current Fund
    p50 = curr_data['P50']
    p10 = curr_data['P10']
    p90 = curr_data['P90']
    
    sigma = (p90 - p10) / 3.29 if p90 != p10 else p50 * 0.1
    x = np.linspace(p10 - sigma, p90 + sigma, 100)
    y = norm.pdf(x, p50, sigma)
    
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"Current: {curr_data['AMC']}", 
                             fill='tozeroy', line=dict(color='#e74c3c', width=2))) # Red

    # 2. Plot Recommended Fund (if different)
    if rec_data and rec_data['AMC'] != curr_data['AMC']:
        p50_r = rec_data['P50']
        p10_r = rec_data['P10']
        p90_r = rec_data['P90']
        
        sigma_r = (p90_r - p10_r) / 3.29 if p90_r != p10_r else p50_r * 0.1
        x_r = np.linspace(p10_r - sigma_r, p90_r + sigma_r, 100)
        y_r = norm.pdf(x_r, p50_r, sigma_r)
        
        fig.add_trace(go.Scatter(x=x_r, y=y_r, mode='lines', name=f"Recommended: {rec_data['AMC']}", 
                                 fill='tozeroy', line=dict(color='#2ecc71', width=2), opacity=0.6)) # Green

    fig.update_layout(title="Probability Distribution Comparison", 
                      xaxis_title="Projected Value (₹)", 
                      yaxis_title="Probability",
                      height=350, margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
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
        # Filter valid schemes from MC data
        schemes = sorted(df_mc['Scheme'].unique())
        selected_scheme = st.selectbox("Select Scheme Category:", schemes)
        
        # Filter valid amounts from MC data
        amounts = sorted(df_mc['SIP_Amount'].unique())
        sip_amount = st.selectbox("Monthly SIP Amount (₹):", amounts, index=amounts.index(5000) if 5000 in amounts else 0)

    with col2:
        duration = st.selectbox("Duration (Months):", [12, 24, 36])
        top_n = st.selectbox("AMCs to be recommended:", [1, 2, 3, 4, 5], index=0)

    if st.button("Generate Recommendation"):
        st.divider()
        
        # Find best funds based on Rank
        ranked_subset = df_ranks[df_ranks['Scheme'] == selected_scheme].sort_values('Final_Score', ascending=False).head(top_n)
        top_amcs = ranked_subset['AMC'].tolist()
        
        best_results = []
        
        for amc in top_amcs:
            invested = sip_amount * duration
            
            if duration == 12:
                # USE FORECAST (12M)
                expected_val = calculate_12m_forecast_sip(amc, selected_scheme, sip_amount)
                p50 = expected_val
                p10 = expected_val * 0.95
                p90 = expected_val * 1.05
            else:
                # USE MONTE CARLO (24/36M)
                row = df_mc[(df_mc['AMC'] == amc) & 
                            (df_mc['Scheme'] == selected_scheme) & 
                            (df_mc['SIP_Amount'] == sip_amount) & 
                            (df_mc['Tenure_Months'] == duration)]
                if not row.empty:
                    p50 = row.iloc[0]['P50_Corpus']
                    p10 = row.iloc[0]['P10_Corpus']
                    p90 = row.iloc[0]['P90_Corpus']
                else:
                    p50, p10, p90 = 0, 0, 0

            if p50 > 0: # Only add if data exists
                best_results.append({
                    "AMC": amc, "Invested": invested, "P50": p50, "P10": p10, "P90": p90
                })
            
        # Display Results
        if best_results:
            top_pick = best_results[0]
            
            st.subheader(f"Best Recommendation: {top_pick['AMC']}")
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Investment", format_currency(top_pick['Invested']))
            m2.metric("Expected Value (P50)", format_currency(top_pick['P50']))
            m3.metric("Optimistic Value (P90)", format_currency(top_pick['P90']))
            m4.metric("Pessimistic Value (P10)", format_currency(top_pick['P10']))
            
            # Bell Curve
            st.plotly_chart(plot_comparison_bell_curve(top_pick), use_container_width=True)
            
            # Table for others
            if len(best_results) > 1:
                st.write(f"### Other Top Recommendations")
                extra_df = pd.DataFrame(best_results[1:])
                # Clean dataframe for display
                display_df = extra_df[['AMC', 'P50', 'P90', 'P10']].copy()
                # Apply formatting safely
                for col in ['P50', 'P90', 'P10']:
                    display_df[col] = display_df[col].apply(lambda x: format_currency(x))
                
                st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No data available for this combination. Please try a different amount or tenure.")

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
            with st.expander(f"Fund {i+1} Details", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                # Dropdowns strictly from valid DF_MC data
                sch = c1.selectbox(f"Scheme", sorted(df_mc['Scheme'].unique()), key=f"s_{i}")
                amc = c2.selectbox(f"AMC", sorted(df_mc['AMC'].unique()), key=f"a_{i}")
                amt = c3.selectbox(f"SIP Amount", sorted(df_mc['SIP_Amount'].unique()), index=9, key=f"m_{i}")
                ten = c4.selectbox(f"Tenure", [12, 24, 36], index=1, key=f"t_{i}")
                user_portfolio.append({'Scheme': sch, 'AMC': amc, 'Amount': amt, 'Tenure': ten})
        
        submitted = st.form_submit_button("Analyze & Rebalance")

    if submitted:
        total_invested = 0
        total_current_p50 = 0
        total_rebal_p50 = 0
        
        st.divider()
        st.markdown("## 📊 Analysis Result")
        
        for idx, fund in enumerate(user_portfolio):
            st.markdown(f"### {idx+1}. {fund['Scheme']} - {fund['AMC']}")
            
            # 1. Get Current Fund Data
            if fund['Tenure'] == 12:
                # 12M Logic
                curr_p50 = calculate_12m_forecast_sip(fund['AMC'], fund['Scheme'], fund['Amount'])
                curr_p10 = curr_p50 * 0.95
                curr_p90 = curr_p50 * 1.05
            else:
                # 24/36M Logic
                curr_row = df_mc[(df_mc['AMC'] == fund['AMC']) & 
                                 (df_mc['Scheme'] == fund['Scheme']) & 
                                 (df_mc['SIP_Amount'] == fund['Amount']) & 
                                 (df_mc['Tenure_Months'] == fund['Tenure'])]
                if not curr_row.empty:
                    curr_p50 = curr_row.iloc[0]['P50_Corpus']
                    curr_p10 = curr_row.iloc[0]['P10_Corpus']
                    curr_p90 = curr_row.iloc[0]['P90_Corpus']
                else:
                    curr_p50 = 0

            # 2. Get Best Fund Data (Recommendation)
            best_amc_name = df_ranks[df_ranks['Scheme'] == fund['Scheme']].sort_values('Final_Score', ascending=False).iloc[0]['AMC']
            
            if fund['Tenure'] == 12:
                best_p50 = calculate_12m_forecast_sip(best_amc_name, fund['Scheme'], fund['Amount'])
                best_p10 = best_p50 * 0.95
                best_p90 = best_p50 * 1.05
            else:
                best_row = df_mc[(df_mc['AMC'] == best_amc_name) & 
                                 (df_mc['Scheme'] == fund['Scheme']) & 
                                 (df_mc['SIP_Amount'] == fund['Amount']) & 
                                 (df_mc['Tenure_Months'] == fund['Tenure'])]
                if not best_row.empty:
                    best_p50 = best_row.iloc[0]['P50_Corpus']
                    best_p10 = best_row.iloc[0]['P10_Corpus']
                    best_p90 = best_row.iloc[0]['P90_Corpus']
                else:
                    best_p50 = 0

            # 3. Calculate Gain & Totals
            if curr_p50 > 0 and best_p50 > 0:
                gain = best_p50 - curr_p50
                total_invested += (fund['Amount'] * fund['Tenure'])
                total_current_p50 += curr_p50
                total_rebal_p50 += best_p50
                
                # Layout: Metrics on Left, Chart on Right
                c_left, c_right = st.columns([1, 2])
                
                with c_left:
                    st.write("**Performance Check**")
                    if gain > 500 and fund['AMC'] != best_amc_name: # Threshold for switching
                        st.error(f"⚠️ Recommendation: **Switch**")
                        st.metric("Potential Gain", format_currency(gain), delta="Rebalance Opportunity")
                        st.write(f"Better Fund: **{best_amc_name}**")
                    else:
                        st.success("✅ Recommendation: **Hold**")
                        st.write("You own the best fund.")
                
                with c_right:
                    # Prepare data for plotting
                    curr_data = {'AMC': fund['AMC'], 'P50': curr_p50, 'P10': curr_p10, 'P90': curr_p90}
                    rec_data = {'AMC': best_amc_name, 'P50': best_p50, 'P10': best_p10, 'P90': best_p90}
                    st.plotly_chart(plot_comparison_bell_curve(curr_data, rec_data), use_container_width=True)
                    
            else:
                st.warning("Insufficient data to analyze this fund combination.")
            
            st.divider()

        # Final Summary
        if total_invested > 0:
            st.markdown("### 🏆 Total Portfolio Impact")
            
            overall_gain = total_rebal_p50 - total_current_p50
            
            col_sum1, col_sum2 = st.columns([1, 2])
            
            with col_sum1:
                st.metric("Total Invested", format_currency(total_invested))
                st.metric("Current Portfolio Value", format_currency(total_current_p50))
                st.metric("Rebalanced Portfolio Value", format_currency(total_rebal_p50), delta=format_currency(overall_gain))
            
            with col_sum2:
                # Summary Bar Chart
                summary_df = pd.DataFrame({
                    "State": ["Current", "Rebalanced"],
                    "Value": [total_current_p50, total_rebal_p50],
                    "Color": ["#95a5a6", "#2ecc71"]
                })
                fig = px.bar(summary_df, x="State", y="Value", text="Value", color="State", 
                             color_discrete_sequence=["#95a5a6", "#2ecc71"], title="Total Wealth Comparison")
                fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
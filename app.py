import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Recommendation", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #2c3e50; }
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
        st.error(f"Missing File: {e}. Please ensure the Excel files are in the same folder as app.py")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_ranks, df_mc, df_forecast = load_data()

# --- HELPER FUNCTIONS ---
def format_currency(value):
    return f"INR {value:,.0f}"

def calculate_12m_forecast_sip(amc, scheme, monthly_sip):
    """Calculates SIP returns using the raw 12-month Forecasted NAV data"""
    subset = df_forecast[(df_forecast['AMC'] == amc) & (df_forecast['Scheme'] == scheme)].sort_values('Date')
    if subset.empty: return 0
    
    # Simulate SIP: Buy units on the 1st of every month for 12 months
    subset['YearMonth'] = subset['Date'].dt.to_period('M')
    monthly_data = subset.groupby('YearMonth').first().head(12) # First 12 months
    
    if len(monthly_data) < 12: return 0
    
    units = 0
    for nav in monthly_data['Forecast_NAV']:
        units += monthly_sip / nav
        
    final_nav = monthly_data.iloc[-1]['Forecast_NAV']
    return units * final_nav

def plot_bell_curve(p10, p50, p90, label):
    """Generates a Bell Curve (Normal Distribution) based on P50 and spread"""
    # Estimate standard deviation (sigma) from P90 and P10 (assuming they are roughly +/- 1.645 sigma)
    # sigma approx = (P90 - P10) / 3.29
    sigma = (p90 - p10) / 3.29 if p90 != p10 else p50 * 0.1 # Fallback sigma
    
    x = np.linspace(p10 - sigma, p90 + sigma, 100)
    y = norm.pdf(x, p50, sigma)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=label, fill='tozeroy', line=dict(color='#4c78a8', width=2)))
    
    # Add markers for P10, P50, P90
    fig.add_trace(go.Scatter(x=[p10, p50, p90], y=[norm.pdf(p10, p50, sigma), norm.pdf(p50, p50, sigma), norm.pdf(p90, p50, sigma)],
                             mode='markers+text', text=['P10', 'P50', 'P90'], textposition="top center",
                             marker=dict(color=['red', 'gold', 'green'], size=10), showlegend=False))
    
    fig.update_layout(title="Projected Probability Distribution (Bell Curve)", 
                      xaxis_title="Projected Corpus Value (INR)", 
                      yaxis_title="Probability Density",
                      height=400, showlegend=False)
    return fig

# --- NAVIGATION ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["First-Time Investor", "Existing Investor"])

# ==========================================
# PAGE: FIRST-TIME INVESTOR
# ==========================================
if page == "First-Time Investor":
    st.title("New Investment Recommendation")
    
    if df_mc.empty: st.stop()

    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Scheme Category
        schemes = sorted(df_mc['Scheme'].unique())
        selected_scheme = st.selectbox("Select Scheme Category:", schemes)
        
        # 2. Monthly SIP (Dropdown)
        amounts = sorted(df_mc['SIP_Amount'].unique())
        sip_amount = st.selectbox("Monthly SIP Amount (INR):", amounts, index=amounts.index(5000) if 5000 in amounts else 0)

    with col2:
        # 3. Duration (Dropdown: 12, 24, 36 only)
        duration = st.selectbox("Duration (Months):", [12, 24, 36])
        
        # 4. Count of AMCs to recommend
        top_n = st.selectbox("AMCs to be recommended:", [1, 2, 3, 4, 5], index=0)

    if st.button("Generate Recommendation"):
        st.divider()
        
        # --- LOGIC SELECTION ---
        # Find best funds in category based on Rank
        ranked_subset = df_ranks[df_ranks['Scheme'] == selected_scheme].sort_values('Final_Score', ascending=False).head(top_n)
        top_amcs = ranked_subset['AMC'].tolist()
        
        best_results = []
        
        for amc in top_amcs:
            invested = sip_amount * duration
            
            if duration == 12:
                # USE FORECAST FILE (Calculated)
                expected_val = calculate_12m_forecast_sip(amc, selected_scheme, sip_amount)
                # Since forecast is single line, estimate P10/P90 for the curve (e.g., +/- 5% spread)
                p50 = expected_val
                p10 = expected_val * 0.95
                p90 = expected_val * 1.05
            else:
                # USE MONTE CARLO FILE (24/36 Months)
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

            best_results.append({
                "AMC": amc,
                "Invested": invested,
                "P50": p50,
                "P10": p10,
                "P90": p90
            })
            
        # Display Results
        if best_results:
            top_pick = best_results[0] # Best ranked
            
            st.subheader(f"Best Recommendation: {top_pick['AMC']}")
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Investment", format_currency(top_pick['Invested']))
            m2.metric("Expected Value (P50)", format_currency(top_pick['P50']))
            m3.metric("Optimistic Value (P90)", format_currency(top_pick['P90']))
            m4.metric("Pessimistic Value (P10)", format_currency(top_pick['P10']))
            
            # Bell Curve
            st.plotly_chart(plot_bell_curve(top_pick['P10'], top_pick['P50'], top_pick['P90'], top_pick['AMC']), use_container_width=True)
            
            # Table for other recommendations if top_n > 1
            if top_n > 1:
                st.write(f"### Other Top {top_n-1} Recommendations")
                extra_df = pd.DataFrame(best_results[1:])
                st.dataframe(extra_df[['AMC', 'P50', 'P90', 'P10']].style.format(format_currency))
        else:
            st.warning("No data available for this combination.")

# ==========================================
# PAGE: EXISTING INVESTOR
# ==========================================
elif page == "Existing Investor":
    st.title("Portfolio Rebalancing")
    
    if df_mc.empty: st.stop()

    # 1. Portfolio Count
    num_funds = st.number_input("Current Portfolio Count:", min_value=1, max_value=10, step=1, value=1)
    
    # 2. Dynamic Inputs
    user_portfolio = []
    st.markdown("### Enter Portfolio Details")
    
    with st.form("portfolio_form"):
        for i in range(num_funds):
            st.markdown(f"**Fund {i+1}**")
            c1, c2, c3, c4 = st.columns(4)
            sch = c1.selectbox(f"Scheme {i+1}", sorted(df_mc['Scheme'].unique()), key=f"s_{i}")
            amc = c2.selectbox(f"AMC {i+1}", sorted(df_mc['AMC'].unique()), key=f"a_{i}")
            amt = c3.selectbox(f"SIP Amount {i+1}", sorted(df_mc['SIP_Amount'].unique()), index=9, key=f"m_{i}") # Default 5000
            ten = c4.selectbox(f"Tenure {i+1}", [12, 24, 36], index=1, key=f"t_{i}") # Default 24
            user_portfolio.append({'Scheme': sch, 'AMC': amc, 'Amount': amt, 'Tenure': ten})
            st.divider()
            
        submitted = st.form_submit_button("Analyze & Rebalance")

    if submitted:
        total_invested = 0
        total_current_p50 = 0
        total_rebal_p50 = 0
        rebal_details = []

        st.markdown("### Analysis & Recommendations")
        
        for fund in user_portfolio:
            # Current Fund Data
            curr_row = df_mc[(df_mc['AMC'] == fund['AMC']) & 
                             (df_mc['Scheme'] == fund['Scheme']) & 
                             (df_mc['SIP_Amount'] == fund['Amount']) & 
                             (df_mc['Tenure_Months'] == fund['Tenure'])]
            
            # Find Best Fund in Category (Using Rank)
            best_in_cat = df_ranks[df_ranks['Scheme'] == fund['Scheme']].sort_values('Final_Score', ascending=False).iloc[0]['AMC']
            
            best_row = df_mc[(df_mc['AMC'] == best_in_cat) & 
                             (df_mc['Scheme'] == fund['Scheme']) & 
                             (df_mc['SIP_Amount'] == fund['Amount']) & 
                             (df_mc['Tenure_Months'] == fund['Tenure'])]

            if not curr_row.empty and not best_row.empty:
                curr_val = curr_row.iloc[0]['P50_Corpus']
                best_val = best_row.iloc[0]['P50_Corpus']
                gain = best_val - curr_val
                
                total_invested += (fund['Amount'] * fund['Tenure'])
                total_current_p50 += curr_val
                total_rebal_p50 += best_val
                
                # Recommendation Logic
                if gain > 0 and fund['AMC'] != best_in_cat:
                    msg = f"Recommendation: Switch to **{best_in_cat}** to potentially gain **{format_currency(gain)}** more."
                    rebal = "Yes"
                else:
                    msg = "Recommendation: **Hold**. You have the best fund."
                    rebal = "No"
                
                rebal_details.append({
                    "Scheme": fund['Scheme'],
                    "Current AMC": fund['AMC'],
                    "Rebalance?": rebal,
                    "Recommendation": msg
                })
            else:
                st.warning(f"Data missing for {fund['AMC']} - {fund['Scheme']}")

        # Display Recommendations Line by Line
        for detail in rebal_details:
            st.info(f"**{detail['Scheme']} ({detail['Current AMC']})**: {detail['Recommendation']}")

        # Final Summary Chart
        st.markdown("### Portfolio Summary")
        
        summary_data = {
            "Category": ["Invested Amount", "Current Portfolio (P50)", "Rebalanced Portfolio (P50)"],
            "Amount": [total_invested, total_current_p50, total_rebal_p50],
            "Color": ["#95a5a6", "#3498db", "#2ecc71"]
        }
        
        fig = px.bar(summary_data, x="Category", y="Amount", text="Amount", color="Category",
                     color_discrete_sequence=summary_data["Color"])
        fig.update_traces(texttemplate='INR %{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, height=400, yaxis_title="Value (INR)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Gain Metric
        total_gain = total_rebal_p50 - total_current_p50
        if total_gain > 0:
            st.success(f"### Total Potential Gain from Rebalancing: {format_currency(total_gain)}")
        else:
            st.success("### Your portfolio is already optimized!")
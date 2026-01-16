import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart SIP Advisor", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 5px; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # 1. Load Rankings
        rankings = pd.read_excel("data/25_Final_AHP_Ranking_5_1.xlsx")
        
        # 2. Load Monte Carlo Results
        mc_results = pd.read_excel("data/26_Monte_Carlo_EWMA_Results.xlsx")
        
        # 3. Load Forecast Data (The 24-Month Predictions)
        forecasts = pd.read_excel("data/13_Forecasted_Fund_NAV.xlsx")
        
        # Clean column names
        rankings.columns = rankings.columns.str.strip()
        mc_results.columns = mc_results.columns.str.strip()
        forecasts.columns = forecasts.columns.str.strip()
        
        # Ensure Date column is datetime
        forecasts['Date'] = pd.to_datetime(forecasts['Date'])
        
        return rankings, mc_results, forecasts
    except FileNotFoundError as e:
        st.error(f"Missing File: {e}. Please check your 'data' folder.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_ranks, df_mc, df_forecast = load_data()

# --- HELPER FUNCTIONS ---
def get_top_recommendations(scheme_name, top_n=3):
    """Find top funds in the category based on AHP Score"""
    if df_ranks.empty: return []
    subset = df_ranks[df_ranks['Scheme'] == scheme_name].sort_values(by='Final_Score', ascending=False)
    return subset.head(top_n)['AMC'].tolist()

def format_currency(value):
    return f"INR {value:,.0f}"

def plot_forecast_trend(amc, scheme):
    """Plots the 24-month CNN-GRU prediction curve"""
    # Filter data for specific AMC and Scheme
    subset = df_forecast[(df_forecast['AMC'] == amc) & (df_forecast['Scheme'] == scheme)]
    
    if subset.empty:
        st.warning(f"No forecast data found for {amc} - {scheme}")
        return

    # Create Line Chart
    fig = px.line(subset, x='Date', y='Forecast_NAV', 
                  title=f"AI Prediction (24 Months): {amc} - {scheme}",
                  labels={'Forecast_NAV': 'Predicted NAV', 'Date': 'Future Date'})
    
    fig.update_traces(line_color='#2ca02c', line_width=3)
    fig.update_layout(height=350, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- NAVIGATION SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "First-Time Investor", "Existing Investor"])

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "Home":
    st.title("Smart SIP Advisor")
    st.markdown("### Data-Driven Portfolio Recommendations")
    st.info("Powered by Hybrid CNN-GRU Forecasting & Monte Carlo Simulations")
    
    st.image("https://images.unsplash.com/photo-1611974765270-ca1258634369?auto=format&fit=crop&q=80&w=1000", caption="AI in Finance", use_container_width=True)

# ==========================================
# PAGE 2: FIRST-TIME INVESTOR
# ==========================================
elif page == "First-Time Investor":
    st.header("New Investor Recommendation")
    
    if df_mc.empty: st.stop()

    # User Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        scheme_list = sorted(df_mc['Scheme'].unique().tolist())
        selected_scheme = st.selectbox("Select Scheme Category", scheme_list)
    with col2:
        valid_amounts = sorted(df_mc['SIP_Amount'].unique().tolist())
        sip_amount = st.select_slider("Monthly SIP Amount (INR)", options=valid_amounts, value=5000)
    with col3:
        valid_tenures = sorted(df_mc['Tenure_Months'].unique().tolist())
        tenure = st.selectbox("Duration (Months)", valid_tenures, index=1)

    if st.button("Get Best Fund"):
        st.divider()
        
        # Logic: Find top ranked funds -> Get their simulation results
        top_amcs = get_top_recommendations(selected_scheme, top_n=5)
        
        results = df_mc[
            (df_mc['Scheme'] == selected_scheme) & 
            (df_mc['SIP_Amount'] == sip_amount) & 
            (df_mc['Tenure_Months'] == tenure) & 
            (df_mc['AMC'].isin(top_amcs))
        ].copy()
        
        # Sort by Ranking Score
        results = pd.merge(results, df_ranks[['AMC', 'Scheme', 'Final_Score']], on=['AMC', 'Scheme'], how='left')
        results = results.sort_values(by='Final_Score', ascending=False)

        if not results.empty:
            best_fund = results.iloc[0]
            
            # --- 1. SHOW RECOMMENDATION ---
            st.success(f"Top Recommendation: **{best_fund['AMC']} {best_fund['Scheme']}**")
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Investment", format_currency(best_fund['Invested']))
            m2.metric("Expected Value (P50)", format_currency(best_fund['P50_Corpus']), delta=f"+{format_currency(best_fund['P50_Profit'])}")
            m3.metric("Optimistic Value (P90)", format_currency(best_fund['P90_Corpus']))

            # --- 2. SHOW AI FORECAST (NEW FEATURE) ---
            st.markdown("### AI Model Forecast (Next 24 Months)")
            st.write(f"This is how our CNN-GRU model predicts the NAV of **{best_fund['AMC']}** will move:")
            plot_forecast_trend(best_fund['AMC'], best_fund['Scheme'])
            
            # --- 3. SHOW MONTE CARLO CHART ---
            st.markdown("### Monte Carlo Simulation Outcomes")
            
            # Bar chart comparing top 3 funds
            top_3 = results.head(3)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=top_3['AMC'], y=top_3['P10_Corpus'], name='Worst Case (P10)', marker_color='red'))
            fig.add_trace(go.Bar(x=top_3['AMC'], y=top_3['P50_Corpus'], name='Expected (P50)', marker_color='gold'))
            fig.add_trace(go.Bar(x=top_3['AMC'], y=top_3['P90_Corpus'], name='Best Case (P90)', marker_color='green'))
            
            fig.update_layout(barmode='group', title="Comparison of Top Recommendations")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("No data found for this combination.")

# ==========================================
# PAGE 3: EXISTING INVESTOR
# ==========================================
elif page == "Existing Investor":
    st.header("Portfolio Rebalancing")
    
    if df_mc.empty: st.stop()

    st.info("Compare your current fund against the AI's top pick.")

    with st.form("existing_form"):
        c1, c2 = st.columns(2)
        curr_amc = c1.selectbox("Current AMC", sorted(df_mc['AMC'].unique()))
        curr_scheme = c2.selectbox("Scheme", sorted(df_mc['Scheme'].unique()))
        curr_sip = c1.number_input("SIP Amount", 500, 20000, 5000, 500)
        curr_tenure = c2.selectbox("Tenure", sorted(df_mc['Tenure_Months'].unique()), index=1)
        
        submitted = st.form_submit_button("Analyze")

    if submitted:
        # Find Current Data
        curr_data = df_mc[(df_mc['AMC']==curr_amc) & (df_mc['Scheme']==curr_scheme) & 
                          (df_mc['SIP_Amount']==curr_sip) & (df_mc['Tenure_Months']==curr_tenure)]
        
        # Find Best Alternative
        candidates = df_mc[(df_mc['Scheme']==curr_scheme) & 
                           (df_mc['SIP_Amount']==curr_sip) & 
                           (df_mc['Tenure_Months']==curr_tenure)].copy()
        candidates = pd.merge(candidates, df_ranks[['AMC', 'Scheme', 'Final_Score']], on=['AMC', 'Scheme'], how='left')
        
        if not candidates.empty and not curr_data.empty:
            best_alt = candidates.sort_values('Final_Score', ascending=False).iloc[0]
            current = curr_data.iloc[0]
            
            st.divider()
            col_res1, col_res2 = st.columns(2)
            
            # Current Fund Column
            with col_res1:
                st.subheader("Your Fund")
                st.write(f"**{curr_amc}**")
                st.metric("Expected Corpus", format_currency(current['P50_Corpus']))
                st.write("**Forecasted Trend:**")
                plot_forecast_trend(curr_amc, curr_scheme)
                
            # Recommended Fund Column
            with col_res2:
                st.subheader("AI Recommended")
                st.write(f"**{best_alt['AMC']}**")
                st.metric("Expected Corpus", format_currency(best_alt['P50_Corpus']), 
                          delta=format_currency(best_alt['P50_Corpus'] - current['P50_Corpus']))
                st.write("**Forecasted Trend:**")
                plot_forecast_trend(best_alt['AMC'], best_alt['Scheme'])
            
            if best_alt['AMC'] != curr_amc:
                st.success(f"Recommendation: Switch to **{best_alt['AMC']}** to potentially gain **{format_currency(best_alt['P50_Corpus'] - current['P50_Corpus'])}** more.")
            else:
                st.success("Great job! You are holding the best fund in this category.")
                
        else:
            st.error("Data not found for the selected inputs.")
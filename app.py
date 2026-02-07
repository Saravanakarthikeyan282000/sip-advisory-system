import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SIP Advisory System", layout="wide")

# --- CUSTOM CSS FOR TRADINGVIEW AESTHETIC ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child {
        background-color: #0088cc; color: white; border-radius: 5px; width: 100%;
        font-weight: bold; border: none;
    }
    h1, h2, h3 { color: #00d4ff !important; font-family: 'Inter', sans-serif; }
    .stMetric { 
        background-color: #161b22; 
        border: 1px solid #30363d; 
        padding: 15px; 
        border-radius: 10px; 
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_data
def load_excel_data():
    try:
        mc_results = pd.read_excel("26_Monte_Carlo_EWMA_Results.xlsx")
        ahp_ranks = pd.read_excel("25_Final_AHP_Ranking_5_1.xlsx")
        ahp_ranks.columns = ahp_ranks.columns.str.strip()
        return mc_results, ahp_ranks
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

mc_results, ahp_ranks = load_excel_data()

if mc_results is not None:
    all_schemes = sorted(ahp_ranks['Scheme'].unique().tolist())
    all_amcs = sorted(mc_results['AMC'].unique().tolist())

    # --- 2. CORE LOGIC ---
    def get_dynamic_best_amcs(scheme, sip_amount, tenure, top_n=1):
        df_filtered = mc_results[(mc_results['Scheme'] == scheme) & 
                                 (mc_results['SIP_Amount'] == sip_amount) & 
                                 (mc_results['Tenure_Months'] == tenure)].copy()
        if df_filtered.empty: return None
        df_ranked = pd.merge(df_filtered, ahp_ranks[['AMC', 'Scheme', 'Final_Score']], on=['AMC', 'Scheme'], how='left')
        df_ranked['Final_Score'] = df_ranked['Final_Score'].fillna(0)
        df_ranked = df_ranked.sort_values(by=['P50_Corpus', 'Final_Score'], ascending=[False, False])
        return df_ranked.head(top_n)

    # --- 3. PROFESSIONAL VISUALIZATIONS (TRADINGVIEW STYLE) ---
    def plot_tradingview_bell_curve(amc, scheme, p10, p50, p90):
        mu, std = p50, (p90 - p10) / 2.56
        x = np.linspace(mu - 4*std, mu + 4*std, 500)
        y = stats.norm.pdf(x, mu, std)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', mode='lines', 
                                 line=dict(color='#00d4ff', width=3),
                                 fillcolor='rgba(0, 212, 255, 0.1)', name='Probability'))

        points, labels, colors = [p10, p50, p90], ['Worst Case', 'Most Likely', 'Best Case'], ['#ff4b4b', '#00d4ff', '#00ff41']

        for p, l, c in zip(points, labels, colors):
            y_val = stats.norm.pdf(p, mu, std)
            fig.add_trace(go.Scatter(x=[p], y=[y_val], mode='markers+text',
                                     text=[f"{l}<br>₹{p:,.0f}"], textposition="top center",
                                     marker=dict(color=c, size=12, symbol='diamond'), showlegend=False))
            fig.add_shape(type="line", x0=p, y0=0, x1=p, y1=y_val, line=dict(color=c, width=1, dash="dot"))

        fig.update_layout(title=f"<b>Wealth Projection: {amc}</b><br><span style='font-size:12px;'>{scheme}</span>",
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            xaxis=dict(title="Corpus Value (₹)", showgrid=False), yaxis=dict(showgrid=False, showticklabels=False))
        return fig

    def plot_tradingview_bar(df_comp):
        categories = ['Existing Strategy', 'Advisory Strategy']
        values = [df_comp['Gain Before Rebalancing'].sum(), df_comp['Gain After Rebalancing'].sum()]
        fig = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=['#44475a', '#00d4ff'],
                                   text=[f"₹{v:,.0f}" for v in values], textposition='auto', width=0.4)])
        fig.update_layout(title="<b>Quantifiable Portfolio Gain Comparison</b>",
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            yaxis=dict(title="Total Gain (₹)", gridcolor="#30363d"), xaxis=dict(showgrid=False))
        return fig

    # --- 4. STREAMLIT UI ---
    st.title(" SIP Advisory System")
    st.divider()

    investor_type = st.sidebar.radio("Investor Profile", ["First-Time Investor", "Existing Investor"])

    if investor_type == "First-Time Investor":
        st.subheader(" Portfolio Configuration Engine")
        c1, c2, c3, c4 = st.columns(4)
        with c1: num_amcs = st.selectbox("No of AMCs", [1,2,3,4,5])
        with c2: sip_amt = st.selectbox("SIP Amount (₹)", [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000], index=2)
        with c3: tenure = st.selectbox("Tenure (Months)", [12, 24, 36], index=2)
        with c4: scheme_choice = st.selectbox("Fund Class", all_schemes)

        if st.button("Generate Portfolio Report"):
            best_df = get_dynamic_best_amcs(scheme_choice, sip_amt, tenure, top_n=num_amcs)
            if best_df is not None:
                numeric_cols = ['Invested', 'P10_Corpus', 'P50_Corpus', 'P90_Corpus']
                format_dict = {col: "₹{:,.0f}" for col in numeric_cols}
                st.dataframe(best_df[['AMC', 'Scheme'] + numeric_cols].style.format(format_dict), use_container_width=True)
                for _, row in best_df.iterrows():
                    st.plotly_chart(plot_tradingview_bell_curve(row['AMC'], row['Scheme'], row['P10_Corpus'], row['P50_Corpus'], row['P90_Corpus']), use_container_width=True)

    else:
        st.subheader("🔁 Strategic Rebalancing Optimizer")
        num_sips = st.sidebar.number_input("Number of Active SIPs", 1, 5, 1)
        input_data = []
        for i in range(num_sips):
            with st.expander(f"SIP #{i+1} Details", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1: amc = st.selectbox(f"AMC", all_amcs, key=f"amc_{i}")
                with c2: sch = st.selectbox(f"Scheme", all_schemes, key=f"sch_{i}")
                with c3: amt = st.selectbox(f"Amount", [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000], index=2, key=f"amt_{i}")
                with c4: ten = st.selectbox(f"Tenure", [12, 24, 36], index=1, key=f"ten_{i}")
                input_data.append({'amc': amc, 'sch': sch, 'amt': amt, 'ten': ten})

        if st.button("Run Rebalancing Algorithm"):
            ex_list, reb_list, comp_list, reb_plot_data = [], [], [], []
            t_cur_p50, t_reb_p50 = 0, 0
            
            for item in input_data:
                # 1. Get Existing Data
                cur = mc_results[(mc_results['AMC']==item['amc']) & (mc_results['Scheme']==item['sch']) & (mc_results['SIP_Amount']==item['amt']) & (mc_results['Tenure_Months']==item['ten'])].iloc[0]
                # 2. Get Best Rebalance Option
                best = get_dynamic_best_amcs(item['sch'], item['amt'], item['ten'], top_n=1).iloc[0]
                
                inv = item['amt'] * item['ten']
                t_cur_p50 += cur['P50_Corpus']
                t_reb_p50 += best['P50_Corpus']
                
                ex_list.append({"Existing AMC": item['amc'], "Current Median": cur['P50_Corpus'], "Action": "🔄 Switch" if best['AMC'] != item['amc'] else "✅ Hold"})
                reb_list.append({"Rebalanced AMC": best['AMC'], "Improvement %": ((best['P50_Corpus'] - cur['P50_Corpus'])/cur['P50_Corpus'])*100})
                comp_list.append({"Gain Before Rebalancing": cur['P50_Corpus'] - inv, "Gain After Rebalancing": best['P50_Corpus'] - inv})
                
                # Store data for Bell Curves
                reb_plot_data.append({'amc': best['AMC'], 'sch': item['sch'], 'p10': best['P10_Corpus'], 'p50': best['P50_Corpus'], 'p90': best['P90_Corpus']})

            # --- Visual Outputs ---
            st.metric("Portfolio Alpha (Rebalanced Gain)", f"₹{(t_reb_p50 - t_cur_p50):,.2f}", delta=f"{((t_reb_p50-t_cur_p50)/t_cur_p50)*100:.2f}%")
            
            col_a, col_b = st.columns(2)
            with col_a: st.write("### Existing Status"); st.table(pd.DataFrame(ex_list))
            with col_b: st.write("### Recommended Status"); st.table(pd.DataFrame(reb_list))
            
            # Gain Comparison Bar Chart
            df_comp = pd.DataFrame(comp_list)
            st.plotly_chart(plot_tradingview_bar(df_comp), use_container_width=True)
            
            # Individual Bell Curves for the Recommended Portfolio
            st.write("### Recommended Fund Projections")
            for plot in reb_plot_data:
                st.plotly_chart(plot_tradingview_bell_curve(plot['amc'], plot['sch'], plot['p10'], plot['p50'], plot['p90']), use_container_width=True)
import os
import streamlit as st
import pandas as pd
from utils.data_fetcher import get_supported_coins, get_market_data
import traceback

# ---- Force Dark Mode ----
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117 !important;
            color: #FAFAFA !important;
        }
        .css-1d391kg, .css-1v0mbdj, .css-1c7y2kd, .css-1lcbmhc {
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }
        .stSidebar, .css-6qob1r {
            background-color: #16171c !important;
        }
        .stButton>button {
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }
        .stDataFrame, .stTable {
            background-color: #0E1117 !important;
            color: #FAFAFA !important;
        }
        /* Optional: style markdown text and headers */
        h1, h2, h3, h4, h5, h6, p, label, .markdown-text-container {
            color: #FAFAFA !important;
        }
    </style>
""", unsafe_allow_html=True)
# ...existing code...

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Crypto Market Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Session State Initialization
# ----------------------------
if 'selected_coins' not in st.session_state:
    st.session_state.selected_coins = ['bitcoin', 'ethereum', 'ripple', 'cardano', 'solana']
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = '30'
if 'compare_coins' not in st.session_state:
    st.session_state.compare_coins = ['bitcoin', 'ethereum']
if 'prediction_coin' not in st.session_state:
    st.session_state.prediction_coin = 'bitcoin'
if 'prediction_days' not in st.session_state:
    st.session_state.prediction_days = 7
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = 'ARIMA'

# ----------------------------
# Title & Description
# ----------------------------
st.title("📊 Cryptocurrency Market Analyzer")
st.markdown("""
This dashboard provides real-time cryptocurrency market analysis, 
historical price tracking, and predictive modeling to help you make informed decisions.
""")

# ----------------------------
# Dashboard Images
# ----------------------------
col1, col2 = st.columns(2)
with col1:
    st.image(
        "https://images.unsplash.com/photo-1554260570-e9689a3418b8",
        caption="Market Analysis Dashboard",
        use_container_width=True
    )
with col2:
    st.image(
        "https://images.unsplash.com/photo-1488459716781-31db52582fe9",
        caption="Market Analytics",
        use_container_width=True
    )

# ----------------------------
# Load Market Data
# ----------------------------
try:
    market_data = get_market_data(st.session_state.selected_coins)
    
    st.subheader("📈 Market Overview")
    
    if not market_data.empty:
        display_df = market_data.copy()
        
        # Format numbers
        display_df['price_change_percentage_24h'] = display_df['price_change_percentage_24h'].apply(
            lambda x: f"{x:.2f}%" if x is not None else "N/A"
        )
        display_df['current_price'] = display_df['current_price'].apply(
            lambda x: f"${x:,.2f}" if x is not None else "N/A"
        )
        display_df['market_cap'] = display_df['market_cap'].apply(
            lambda x: f"${x:,.0f}" if x is not None else "N/A"
        )
        display_df['total_volume'] = display_df['total_volume'].apply(
            lambda x: f"${x:,.0f}" if x is not None else "N/A"
        )
        
        display_df = display_df.rename(columns={
            'id': 'ID',
            'symbol': 'Symbol',
            'name': 'Name',
            'current_price': 'Current Price (USD)',
            'market_cap': 'Market Cap (USD)',
            'market_cap_rank': 'Market Cap Rank',
            'price_change_percentage_24h': '24h Change',
            'total_volume': '24h Volume (USD)'
        })
        
        st.dataframe(
            display_df[['Symbol', 'Name', 'Current Price (USD)', '24h Change', 'Market Cap (USD)', 'Market Cap Rank', '24h Volume (USD)']].style.set_table_styles([
                {"selector": "thead th", "props": [("background-color", "#262730"), ("color", "#FAFAFA")]},
                {"selector": "tbody td", "props": [("background-color", "#0E1117"), ("color", "#FAFAFA")]}
            ]),
            use_container_width=True
        )
    else:
        st.error("⚠️ No market data available. Please check your connection or selected cryptocurrencies.")
    
    # ----------------------------
    # Navigation Buttons
    # ----------------------------
    st.subheader("📂 Navigate to Detailed Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📈 Market Overview", use_container_width=True):
            st.switch_page("pages/market_overview.py")
    with col2:
        if st.button("🔍 Cryptocurrency Detail", use_container_width=True):
            st.switch_page("pages/cryptocurrency_detail.py")
    with col3:
        if st.button("🔮 Price Prediction", use_container_width=True):
            st.switch_page("pages/price_prediction.py")
    with col4:
        if st.button("⚖️ Compare Cryptocurrencies", use_container_width=True):
            st.switch_page("pages/compare_cryptos.py")
    
except Exception as e:
    st.error(f"❌ Error loading market data: {str(e)}")
    st.info("Please check your internet connection and try again.")

# ----------------------------
# Cryptocurrency Charts
# ----------------------------
st.subheader("📊 Cryptocurrency Market Charts")
col1, col2 = st.columns(2)
with col1:
    st.image(
        "https://images.unsplash.com/photo-1639754390580-2e7437267698",
        caption="Cryptocurrency Trading", use_container_width=True
    )
    st.image(
        "https://images.unsplash.com/photo-1639987402632-d7273e921454",
        caption="Market Analysis Chart", use_container_width=True
    )
with col2:
    st.image(
        "https://images.unsplash.com/photo-1639389016105-2fb11199fb6b",
        caption="Cryptocurrency Dashboard", use_container_width=True
    )
    st.image(
        "https://images.unsplash.com/photo-1640592276475-56a1c277a38f",
        caption="Crypto Trading Analytics", use_container_width=True
    )

# ----------------------------
# Sidebar Settings
# ----------------------------
st.sidebar.title("⚙️ Dashboard Settings")

try:
    all_coins = get_supported_coins()
    
    if all_coins:
        st.sidebar.subheader("Select Cryptocurrencies")
        selected_coins = st.sidebar.multiselect(
            "Choose coins to track:",
            options=all_coins,
            default=st.session_state.selected_coins,
            format_func=lambda x: x.title()
        )
        if selected_coins:
            st.session_state.selected_coins = selected_coins
        
        # Timeframe selection
        timeframe_options = {
            '1': 'Last 24 Hours',
            '7': 'Last 7 Days',
            '30': 'Last 30 Days',
            '90': 'Last 90 Days',
            '365': 'Last Year',
            'max': 'Maximum Available'
        }
        selected_timeframe = st.sidebar.selectbox(
            "Select timeframe for analysis:",
            options=list(timeframe_options.keys()),
            format_func=lambda x: timeframe_options[x],
            index=list(timeframe_options.keys()).index(st.session_state.timeframe)
        )
        st.session_state.timeframe = selected_timeframe
        
        # About section
        st.sidebar.subheader("ℹ️ About")
        st.sidebar.info(
            "This Cryptocurrency Market Analyzer provides real-time market data, "
            "historical price charts, and predictive modeling to help you make "
            "informed investment decisions.\n\nData is sourced from CoinGecko API."
        )
    else:
        st.sidebar.error("⚠️ Could not load supported cryptocurrencies.")
except Exception as e:
    st.sidebar.error(f"❌ Error: {str(e)}")
    st.sidebar.info("Using default coin selection.")

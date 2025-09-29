import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import streamlit as st

COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
CACHE_TIME = 300  # 5 minutes

# --- Rate Limiting Function ---
def safe_api_call(url, params=None):
    if not hasattr(safe_api_call, "_last_call"):
        safe_api_call._last_call = 0
    now = time.time()
    elapsed = now - safe_api_call._last_call
    delay = 2.1  # slightly above 2 seconds for safety (28 calls/min)
    if elapsed < delay:
        time.sleep(delay - elapsed)
    safe_api_call._last_call = time.time()
    try:
        return requests.get(url, params=params)
    except Exception as e:
        st.error(f"Error making API request: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TIME)
def get_supported_coins():
    try:
        response = safe_api_call(f"{COINGECKO_API_URL}/coins/list")
        if response and response.status_code == 200:
            # Limit coins to popular ones for demo/quick start
            popular_coins = [
                'bitcoin', 'ethereum', 'ripple', 'cardano', 'solana', 
                'polkadot', 'dogecoin', 'avalanche-2', 'chainlink',
                'litecoin', 'polygon', 'binancecoin', 'uniswap',
                'tron', 'stellar', 'cosmos', 'monero', 'algorand', 'tezos'
            ]
            return popular_coins
        else:
            return None
    except Exception as e:
        st.error(f"Error getting supported coins: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TIME)
def get_market_data(coin_ids):
    try:
        params = {
            'ids': ','.join(coin_ids),
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 100,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '24h'
        }
        response = safe_api_call(f"{COINGECKO_API_URL}/coins/markets", params=params)
        if response and response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TIME)
def get_historical_prices(coin_id, days='30', vs_currency='usd'):
    try:
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        response = safe_api_call(f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart", params=params)
        if response and response.status_code == 200:
            data = response.json()
            df_price = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df_price['timestamp'] = pd.to_datetime(df_price['timestamp'], unit='ms')
            df_volume = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            df_volume['timestamp'] = pd.to_datetime(df_volume['timestamp'], unit='ms')
            df_market_cap = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
            df_market_cap['timestamp'] = pd.to_datetime(df_market_cap['timestamp'], unit='ms')
            df = pd.merge(df_price, df_volume, on='timestamp')
            df = pd.merge(df, df_market_cap, on='timestamp')
            df = df.set_index('timestamp')
            return df
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TIME)
def get_coin_details(coin_id):
    try:
        response = safe_api_call(
            f"{COINGECKO_API_URL}/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
        )
        if response and response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error fetching coin details: {str(e)}")
        return {}

def export_data_to_csv(data, filename):
    try:
        return data.to_csv().encode('utf-8')
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
        return

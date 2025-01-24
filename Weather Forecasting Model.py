pip install requests pandas numpy statsmodels matplotlib schedule
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import schedule
import time

# === Weather Data (OpenWeatherMap) ===
def fetch_weather_data():
    API_KEY = 'your_openweather_api_key'  # Replace with your API key
    CITY = 'New York'
    LAT, LON = 40.7128, -74.0060  # Latitude and Longitude for New York
    BASE_URL = 'https://api.openweathermap.org/data/2.5/onecall/timemachine'

    # Get historical data for 30 days ago
    date = datetime.now().timestamp() - (30 * 24 * 60 * 60)  # 30 days ago
    url = f"{BASE_URL}?lat={LAT}&lon={LON}&dt={int(date)}&appid={API_KEY}"

    response = requests.get(url)
    data = response.json()

    # Extract relevant weather data
    weather_data = {
        'date': [datetime.utcfromtimestamp(item['dt']).strftime('%Y-%m-%d %H:%M:%S') for item in data['hourly']],
        'temperature': [item['temp'] - 273.15 for item in data['hourly']],  # Convert from Kelvin to Celsius
    }

    # Convert to DataFrame
    df_weather = pd.DataFrame(weather_data)
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    df_weather.set_index('date', inplace=True)

    return df_weather

# === Air Quality Data (OpenAQ) ===
def fetch_air_quality_data():
    API_URL = 'https://api.openaq.org/v1/measurements'
    response = requests.get(API_URL, params={
        'city': 'Los Angeles',
        'parameter': 'pm25',  # PM2.5 particulate matter
        'date_gte': '2024-01-01',
        'date_lte': '2024-01-30'
    })

    data = response.json()
    air_quality_data = {
        'date': [item['date']['utc'] for item in data['results']],
        'pm25': [item['value'] for item in data['results']],
    }

    # Convert to DataFrame
    df_air_quality = pd.DataFrame(air_quality_data)
    df_air_quality['date'] = pd.to_datetime(df_air_quality['date'])
    df_air_quality.set_index('date', inplace=True)

    return df_air_quality

# === Financial Market Data (Alpha Vantage) ===
def fetch_stock_data():
    API_KEY = 'your_alpha_vantage_api_key'  # Replace with your Alpha Vantage API key
    SYMBOL = 'AAPL'  # Example: Apple stock symbol
    TIME_SERIES = 'TIME_SERIES_DAILY'

    # Construct API URL
    url = f'https://www.alphavantage.co/query?function={TIME_SERIES}&symbol={SYMBOL}&apikey={API_KEY}'

    response = requests.get(url)
    data = response.json()

    # Extract stock price data
    stock_data = {
        'date': [key for key in data['Time Series (Daily)'].keys()],
        'close': [value['4. close'] for value in data['Time Series (Daily)'].values()]
    }

    # Convert to DataFrame
    df_stock = pd.DataFrame(stock_data)
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_stock.set_index('date', inplace=True)

    return df_stock

# === Forecasting with Holt-Winters Exponential Smoothing ===
def forecast_time_series(df, title):
    # Apply Holt-Winters Exponential Smoothing for forecasting
    model = ExponentialSmoothing(df, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=30)  # Forecast next 30 time points
    forecast_df = pd.DataFrame(forecast, columns=['forecast'], index=pd.date_range(df.index[-1], periods=31, freq='D')[1:])

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(df, label='Historical Data')
    plt.plot(forecast_df, label='Forecast', linestyle='--')
    plt.title(f'{title} Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    return forecast_df

# === Automated Data Fetching and Forecasting ===
def run_forecasting():
    # Fetch data from each domain
    df_weather = fetch_weather_data()
    df_air_quality = fetch_air_quality_data()
    df_stock = fetch_stock_data()

    # Perform forecasting for each dataset
    print("\nWeather Data Forecast:")
    forecast_weather = forecast_time_series(df_weather['temperature'], 'Weather Temperature')

    print("\nAir Quality Data Forecast:")
    forecast_air_quality = forecast_time_series(df_air_quality['pm25'], 'Air Quality PM2.5')

    print("\nStock Data Forecast:")
    forecast_stock = forecast_time_series(df_stock['close'], 'Stock Price')

    # Return all forecasts for further analysis or reporting
    return forecast_weather, forecast_air_quality, forecast_stock

# === Scheduling Automated Task ===
def scheduled_task():
    print("Fetching and forecasting data...")
    run_forecasting()

# Schedule to run the task every day at 8 AM
schedule.every().day.at("08:00").do(scheduled_task)

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(60)  # wait for the next minute


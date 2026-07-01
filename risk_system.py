"""
Disaster Risk Assessment System - Nepal Districts
Complete 77 Districts + Interactive NASA POWER Visualizations
"""

import streamlit as st
import google.generativeai as genai
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Nepal Disaster Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - PROFESSIONAL UI/UX
# ============================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .risk-low { background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); border-radius: 15px; padding: 1.5rem; text-align: center; color: #1a472a; }
    .risk-moderate { background: linear-gradient(135deg, #ffe259 0%, #ffa751 100%); border-radius: 15px; padding: 1.5rem; text-align: center; color: #5c3d00; }
    .risk-high { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); border-radius: 15px; padding: 1.5rem; text-align: center; color: #8b0000; }
    .risk-severe { background: linear-gradient(135deg, #f43b47 0%, #ff4d4d 100%); border-radius: 15px; padding: 1.5rem; text-align: center; color: white; animation: pulse 2s infinite; }
    
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
    
    .hazard-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-left: 5px solid;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .hazard-card:hover { transform: translateX(5px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    
    .ai-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1.8rem;
        color: white;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIGURATION
# ============================================
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
MODEL_PATH = "Regression_model.joblib"

# Only configure Gemini if API key is provided
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# ============================================
# ALL 77 DISTRICTS OF NEPAL (COMPLETE)
# ============================================
NEPAL_DISTRICTS = {
    # Province 1 (14 districts)
    'Bhojpur': {'lat': 27.1670, 'lon': 87.0500, 'province': 'Province 1', 'region': 'Hilly'},
    'Dhankuta': {'lat': 26.9833, 'lon': 87.3333, 'province': 'Province 1', 'region': 'Hilly'},
    'Ilam': {'lat': 26.9167, 'lon': 87.9333, 'province': 'Province 1', 'region': 'Hilly'},
    'Jhapa': {'lat': 26.5500, 'lon': 88.0833, 'province': 'Province 1', 'region': 'Terai'},
    'Khotang': {'lat': 27.1667, 'lon': 86.7667, 'province': 'Province 1', 'region': 'Hilly'},
    'Morang': {'lat': 26.4833, 'lon': 87.3333, 'province': 'Province 1', 'region': 'Terai'},
    'Okhaldhunga': {'lat': 27.3167, 'lon': 86.5000, 'province': 'Province 1', 'region': 'Hilly'},
    'Panchthar': {'lat': 27.0833, 'lon': 87.8000, 'province': 'Province 1', 'region': 'Hilly'},
    'Sankhuwasabha': {'lat': 27.3333, 'lon': 87.2500, 'province': 'Province 1', 'region': 'Mountain'},
    'Solukhumbu': {'lat': 27.5833, 'lon': 86.7000, 'province': 'Province 1', 'region': 'Mountain'},
    'Sunsari': {'lat': 26.6333, 'lon': 87.1667, 'province': 'Province 1', 'region': 'Terai'},
    'Taplejung': {'lat': 27.3500, 'lon': 87.6667, 'province': 'Province 1', 'region': 'Mountain'},
    'Terhathum': {'lat': 27.2500, 'lon': 87.5667, 'province': 'Province 1', 'region': 'Hilly'},
    'Udayapur': {'lat': 26.9167, 'lon': 86.8333, 'province': 'Province 1', 'region': 'Hilly'},
    
    # Province 2 (8 districts)
    'Bara': {'lat': 27.0000, 'lon': 85.0000, 'province': 'Province 2', 'region': 'Terai'},
    'Dhanusa': {'lat': 26.8167, 'lon': 86.0000, 'province': 'Province 2', 'region': 'Terai'},
    'Mahottari': {'lat': 26.8333, 'lon': 85.8000, 'province': 'Province 2', 'region': 'Terai'},
    'Parsa': {'lat': 27.3333, 'lon': 84.7500, 'province': 'Province 2', 'region': 'Terai'},
    'Rautahat': {'lat': 26.9000, 'lon': 85.2667, 'province': 'Province 2', 'region': 'Terai'},
    'Saptari': {'lat': 26.5833, 'lon': 86.7500, 'province': 'Province 2', 'region': 'Terai'},
    'Sarlahi': {'lat': 26.9167, 'lon': 85.5333, 'province': 'Province 2', 'region': 'Terai'},
    'Siraha': {'lat': 26.6500, 'lon': 86.2000, 'province': 'Province 2', 'region': 'Terai'},
    
    # Bagmati Province (13 districts)
    'Bhaktapur': {'lat': 27.6667, 'lon': 85.4167, 'province': 'Bagmati', 'region': 'Hilly'},
    'Chitwan': {'lat': 27.6423, 'lon': 84.4142, 'province': 'Bagmati', 'region': 'Terai'},
    'Dhading': {'lat': 27.9167, 'lon': 84.9000, 'province': 'Bagmati', 'region': 'Hilly'},
    'Dolakha': {'lat': 27.7500, 'lon': 86.1833, 'province': 'Bagmati', 'region': 'Hilly'},
    'Kathmandu': {'lat': 27.7172, 'lon': 85.3240, 'province': 'Bagmati', 'region': 'Hilly'},
    'Kavrepalanchok': {'lat': 27.5833, 'lon': 85.5500, 'province': 'Bagmati', 'region': 'Hilly'},
    'Lalitpur': {'lat': 27.6667, 'lon': 85.3167, 'province': 'Bagmati', 'region': 'Hilly'},
    'Makwanpur': {'lat': 27.4167, 'lon': 85.0833, 'province': 'Bagmati', 'region': 'Hilly'},
    'Nuwakot': {'lat': 27.9167, 'lon': 85.2500, 'province': 'Bagmati', 'region': 'Hilly'},
    'Ramechhap': {'lat': 27.5000, 'lon': 86.0833, 'province': 'Bagmati', 'region': 'Hilly'},
    'Rasuwa': {'lat': 28.0667, 'lon': 85.3667, 'province': 'Bagmati', 'region': 'Mountain'},
    'Sindhuli': {'lat': 27.2500, 'lon': 85.9667, 'province': 'Bagmati', 'region': 'Hilly'},
    'Sindhupalchok': {'lat': 27.9833, 'lon': 85.6667, 'province': 'Bagmati', 'region': 'Hilly'},
    
    # Gandaki Province (11 districts)
    'Baglung': {'lat': 28.2667, 'lon': 83.6000, 'province': 'Gandaki', 'region': 'Hilly'},
    'Gorkha': {'lat': 28.0000, 'lon': 84.6333, 'province': 'Gandaki', 'region': 'Hilly'},
    'Kaski': {'lat': 28.2096, 'lon': 83.9856, 'province': 'Gandaki', 'region': 'Hilly'},
    'Lamjung': {'lat': 28.2000, 'lon': 84.3833, 'province': 'Gandaki', 'region': 'Hilly'},
    'Manang': {'lat': 28.6667, 'lon': 84.2500, 'province': 'Gandaki', 'region': 'Mountain'},
    'Mustang': {'lat': 28.8333, 'lon': 83.8333, 'province': 'Gandaki', 'region': 'Mountain'},
    'Myagdi': {'lat': 28.5000, 'lon': 83.5000, 'province': 'Gandaki', 'region': 'Hilly'},
    'Nawalpur': {'lat': 27.6833, 'lon': 84.1167, 'province': 'Gandaki', 'region': 'Terai'},
    'Parbat': {'lat': 28.2333, 'lon': 83.7000, 'province': 'Gandaki', 'region': 'Hilly'},
    'Syangja': {'lat': 28.0833, 'lon': 83.8667, 'province': 'Gandaki', 'region': 'Hilly'},
    'Tanahu': {'lat': 27.9167, 'lon': 84.2500, 'province': 'Gandaki', 'region': 'Hilly'},
    
    # Lumbini Province (12 districts)
    'Arghakhanchi': {'lat': 27.9167, 'lon': 83.0833, 'province': 'Lumbini', 'region': 'Hilly'},
    'Banke': {'lat': 28.0504, 'lon': 81.6163, 'province': 'Lumbini', 'region': 'Terai'},
    'Bardiya': {'lat': 28.3000, 'lon': 81.3667, 'province': 'Lumbini', 'region': 'Terai'},
    'Dang': {'lat': 28.0000, 'lon': 82.3000, 'province': 'Lumbini', 'region': 'Terai'},
    'Gulmi': {'lat': 28.0667, 'lon': 83.2500, 'province': 'Lumbini', 'region': 'Hilly'},
    'Kapilvastu': {'lat': 27.5333, 'lon': 83.0500, 'province': 'Lumbini', 'region': 'Terai'},
    'Palpa': {'lat': 27.9167, 'lon': 83.5500, 'province': 'Lumbini', 'region': 'Hilly'},
    'Parasi': {'lat': 27.6333, 'lon': 83.6500, 'province': 'Lumbini', 'region': 'Terai'},
    'Pyuthan': {'lat': 28.0833, 'lon': 82.8333, 'province': 'Lumbini', 'region': 'Hilly'},
    'Rolpa': {'lat': 28.3667, 'lon': 82.7500, 'province': 'Lumbini', 'region': 'Hilly'},
    'Rukum East': {'lat': 28.6333, 'lon': 82.8333, 'province': 'Lumbini', 'region': 'Hilly'},
    'Rupandehi': {'lat': 27.5000, 'lon': 83.4500, 'province': 'Lumbini', 'region': 'Terai'},
    
    # Karnali Province (10 districts)
    'Dailekh': {'lat': 28.8333, 'lon': 81.7000, 'province': 'Karnali', 'region': 'Hilly'},
    'Dolpa': {'lat': 28.9333, 'lon': 82.9167, 'province': 'Karnali', 'region': 'Mountain'},
    'Humla': {'lat': 29.9667, 'lon': 81.8333, 'province': 'Karnali', 'region': 'Mountain'},
    'Jajarkot': {'lat': 28.7000, 'lon': 82.3000, 'province': 'Karnali', 'region': 'Hilly'},
    'Jumla': {'lat': 29.2667, 'lon': 82.1667, 'province': 'Karnali', 'region': 'Mountain'},
    'Kalikot': {'lat': 29.1333, 'lon': 81.7500, 'province': 'Karnali', 'region': 'Mountain'},
    'Mugu': {'lat': 29.5500, 'lon': 82.2333, 'province': 'Karnali', 'region': 'Mountain'},
    'Salyan': {'lat': 28.3667, 'lon': 82.1667, 'province': 'Karnali', 'region': 'Hilly'},
    'Surkhet': {'lat': 28.6000, 'lon': 81.6167, 'province': 'Karnali', 'region': 'Hilly'},
    'Western Rukum': {'lat': 28.6333, 'lon': 82.8333, 'province': 'Karnali', 'region': 'Hilly'},
    
    # Sudurpashchim Province (9 districts)
    'Achham': {'lat': 29.0000, 'lon': 81.2500, 'province': 'Sudurpashchim', 'region': 'Hilly'},
    'Baitadi': {'lat': 29.5000, 'lon': 80.5000, 'province': 'Sudurpashchim', 'region': 'Hilly'},
    'Bajhang': {'lat': 29.5333, 'lon': 81.2000, 'province': 'Sudurpashchim', 'region': 'Mountain'},
    'Bajura': {'lat': 29.4500, 'lon': 81.6667, 'province': 'Sudurpashchim', 'region': 'Mountain'},
    'Dadeldhura': {'lat': 29.3000, 'lon': 80.5833, 'province': 'Sudurpashchim', 'region': 'Hilly'},
    'Darchula': {'lat': 29.8333, 'lon': 80.5333, 'province': 'Sudurpashchim', 'region': 'Mountain'},
    'Doti': {'lat': 29.2667, 'lon': 80.9833, 'province': 'Sudurpashchim', 'region': 'Hilly'},
    'Kailali': {'lat': 28.6999, 'lon': 80.5961, 'province': 'Sudurpashchim', 'region': 'Terai'},
    'Kanchanpur': {'lat': 28.8333, 'lon': 80.3333, 'province': 'Sudurpashchim', 'region': 'Terai'},
}

# Sort districts alphabetically
NEPAL_DISTRICTS = dict(sorted(NEPAL_DISTRICTS.items()))

# ============================================
# NASA POWER API WITH RETRY LOGIC & VALIDATION
# ============================================

class NASAPowerAPI:
    def __init__(self):
        self.base_url = NASA_POWER_BASE_URL
        self.cache = {}
    
    def _is_valid_value(self, value):
        """Check if value is valid (not NASA's -999 fill value)"""
        if value is None:
            return False
        if isinstance(value, (int, float)):
            # NASA uses -999 as fill value for missing data
            return value != -999 and abs(value) < 1000
        return True
    
    def _get_safe_value(self, data_dict, key, default=0):
        """Safely get value, handling NASA's -999 fill values"""
        value = data_dict.get(key, default)
        if not self._is_valid_value(value):
            return default
        return value
    
    def get_current_weather(self, district_name, retry=3):
        """Fetch weather data with retry logic and validation"""
        if district_name not in NEPAL_DISTRICTS:
            return self._get_mock_weather(district_name)
        
        # Check cache (5 minute TTL)
        if district_name in self.cache:
            cached_time, cached_data = self.cache[district_name]
            if (datetime.now() - cached_time).seconds < 300:
                return cached_data
        
        coords = NEPAL_DISTRICTS[district_name]
        
        # Fetch last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        params = {
            'parameters': 'T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,WS10M,PRECTOTCORR,PS,QV2M',
            'community': 'RE',
            'longitude': coords['lon'],
            'latitude': coords['lat'],
            'start': start_date.strftime('%Y%m%d'),
            'end': end_date.strftime('%Y%m%d'),
            'format': 'JSON',
            'header': 'true'
        }
        
        for attempt in range(retry):
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    properties = data['properties']['parameter']
                    
                    # Get all dates sorted
                    all_dates = sorted(properties['T2M'].keys())
                    
                    if not all_dates:
                        return self._get_mock_weather(district_name)
                    
                    latest_date = all_dates[-1]
                    
                    # SAFELY get values with validation
                    temp_today = self._get_safe_value(properties['T2M'], latest_date, 22.0)
                    temp_max = self._get_safe_value(properties['T2M_MAX'], latest_date, temp_today + 3)
                    temp_min = self._get_safe_value(properties['T2M_MIN'], latest_date, temp_today - 3)
                    humidity = self._get_safe_value(properties['RH2M'], latest_date, 65.0)
                    wind_speed = self._get_safe_value(properties['WS2M'], latest_date, 5.0) * 3.6
                    wind_speed_10m = self._get_safe_value(properties['WS10M'], latest_date, 7.0) * 3.6
                    rainfall_today = max(0, self._get_safe_value(properties['PRECTOTCORR'], latest_date, 0.0))
                    pressure = self._get_safe_value(properties['PS'], latest_date, 1013.0)
                    specific_humidity = self._get_safe_value(properties['QV2M'], latest_date, 10.0)
                    
                    # Calculate 7-day and 30-day averages (only using valid values)
                    rain_7day = 0
                    rain_30day = 0
                    temp_7day_sum = 0
                    valid_days_7 = 0
                    valid_days_30 = 0
                    
                    for date in all_dates[-7:]:
                        rain = self._get_safe_value(properties['PRECTOTCORR'], date, 0)
                        temp = self._get_safe_value(properties['T2M'], date, temp_today)
                        if rain >= 0:
                            rain_7day += rain
                            valid_days_7 += 1
                        if abs(temp) < 100:
                            temp_7day_sum += temp
                    
                    for date in all_dates[-30:]:
                        rain = self._get_safe_value(properties['PRECTOTCORR'], date, 0)
                        if rain >= 0:
                            rain_30day += rain
                            valid_days_30 += 1
                    
                    rain_7day = rain_7day if valid_days_7 > 0 else rainfall_today * 7
                    rain_30day = rain_30day if valid_days_30 > 0 else rainfall_today * 30
                    temp_7day_avg = temp_7day_sum / max(valid_days_7, 1)
                    
                    # Get historical data for chart (only valid values)
                    historical = []
                    for date in all_dates[-30:]:
                        hist_temp = self._get_safe_value(properties['T2M'], date, 20.0)
                        hist_humidity = self._get_safe_value(properties['RH2M'], date, 60.0)
                        hist_rainfall = max(0, self._get_safe_value(properties['PRECTOTCORR'], date, 0.0))
                        hist_wind = self._get_safe_value(properties['WS2M'], date, 5.0) * 3.6
                        
                        # Only add if temp is reasonable (not -999)
                        if abs(hist_temp) < 100:
                            historical.append({
                                'date': datetime.strptime(date, '%Y%m%d'),
                                'temperature': hist_temp,
                                'humidity': min(100, max(0, hist_humidity)),
                                'rainfall': hist_rainfall,
                                'wind_speed': min(150, max(0, hist_wind))
                            })
                    
                    # If no valid historical data, generate mock data
                    if len(historical) < 7:
                        return self._get_mock_weather(district_name)
                    
                    weather = {
                        'temperature': round(temp_today, 1),
                        'temperature_max': round(temp_max, 1),
                        'temperature_min': round(temp_min, 1),
                        'humidity': round(min(100, max(0, humidity)), 1),
                        'wind_speed': round(min(150, max(0, wind_speed)), 1),
                        'wind_speed_10m': round(min(150, max(0, wind_speed_10m)), 1),
                        'rainfall_today': round(rainfall_today, 1),
                        'rainfall_7day': round(rain_7day, 1),
                        'rainfall_30day': round(rain_30day, 1),
                        'temp_7day_avg': round(temp_7day_avg, 1),
                        'pressure': round(pressure, 1),
                        'specific_humidity': round(specific_humidity, 2),
                        'data_source': 'NASA POWER Satellite',
                        'date': latest_date,
                        'province': NEPAL_DISTRICTS[district_name]['province'],
                        'region': NEPAL_DISTRICTS[district_name]['region'],
                        'historical': historical
                    }
                    weather['hazards'] = self._assess_hazards(weather)
                    
                    # Cache the result
                    self.cache[district_name] = (datetime.now(), weather)
                    return weather
                else:
                    if attempt == retry - 1:
                        return self._get_mock_weather(district_name)
                    time.sleep(1)
            except Exception as e:
                print(f"Error fetching weather for {district_name}: {e}")
                if attempt == retry - 1:
                    return self._get_mock_weather(district_name)
                time.sleep(1)
        
        return self._get_mock_weather(district_name)
    
    def _assess_hazards(self, weather):
        hazards = []
        
        # Flood risk based on rainfall
        if weather['rainfall_today'] > 50:
            hazards.append({'type': '🌊 SEVERE FLOOD', 'severity': 'HIGH', 
                           'info': f"{weather['rainfall_today']}mm rain today - Major flooding expected"})
        elif weather['rainfall_today'] > 30:
            hazards.append({'type': '🌊 FLOOD WARNING', 'severity': 'HIGH', 
                           'info': f"{weather['rainfall_today']}mm rain - Flooding possible in low areas"})
        elif weather['rainfall_7day'] > 100:
            hazards.append({'type': '🌊 FLOOD WATCH', 'severity': 'MODERATE', 
                           'info': f"{weather['rainfall_7day']}mm in 7 days - Saturated soil"})
        
        # Heat wave
        if weather['temperature'] > 35:
            hazards.append({'type': '🔥 EXTREME HEAT', 'severity': 'HIGH', 
                           'info': f"{weather['temperature']}°C - Dangerous heat levels"})
        elif weather['temperature'] > 32:
            hazards.append({'type': '🔥 HEAT WAVE', 'severity': 'MODERATE', 
                           'info': f"{weather['temperature']}°C - Stay hydrated"})
        
        # Wind warning
        if weather['wind_speed'] > 50:
            hazards.append({'type': '💨 STORM FORCE', 'severity': 'HIGH', 
                           'info': f"{weather['wind_speed']}km/h winds - Structural damage risk"})
        elif weather['wind_speed'] > 35:
            hazards.append({'type': '💨 STRONG WIND', 'severity': 'MODERATE', 
                           'info': f"{weather['wind_speed']}km/h winds - Secure outdoor items"})
        
        # Landslide risk
        if weather['region'] in ['Hilly', 'Mountain'] and weather['rainfall_today'] > 40:
            hazards.append({'type': '⛰️ LANDSLIDE RISK', 'severity': 'HIGH', 
                           'info': 'Heavy rain on slopes - Potential landslides'})
        elif weather['region'] in ['Hilly', 'Mountain'] and weather['rainfall_7day'] > 80:
            hazards.append({'type': '⛰️ LANDSLIDE WATCH', 'severity': 'MODERATE', 
                           'info': 'Saturated soil - Monitor slopes'})
        
        # Cold wave (winter months)
        current_month = datetime.now().month
        if weather['temperature'] < 5 and current_month in [12, 1, 2]:
            hazards.append({'type': '❄️ COLD WAVE', 'severity': 'HIGH', 
                           'info': f"{weather['temperature']}°C - Risk of hypothermia"})
        
        # Fog (Terai in winter)
        if weather['region'] == 'Terai' and current_month in [12, 1] and weather['humidity'] > 85:
            hazards.append({'type': '🌫️ DENSE FOG', 'severity': 'MODERATE', 
                           'info': 'Reduced visibility - Travel delays'})
        
        if not hazards:
            hazards.append({'type': '✅ NORMAL', 'severity': 'LOW', 'info': 'No active hazards'})
        
        return hazards
    
    def _get_mock_weather(self, district_name):
        """Generate realistic mock weather data"""
        historical = []
        base_temp = 22
        base_humidity = 65
        
        for i in range(30):
            date = datetime.now() - timedelta(days=29-i)
            historical.append({
                'date': date,
                'temperature': round(base_temp + np.random.randn() * 3, 1),
                'humidity': round(min(95, max(40, base_humidity + np.random.randn() * 10)), 1),
                'rainfall': round(max(0, 5 + np.random.randn() * 15), 1),
                'wind_speed': round(max(0, 8 + np.random.randn() * 5), 1)
            })
        
        latest = historical[-1]
        
        return {
            'temperature': latest['temperature'],
            'temperature_max': latest['temperature'] + np.random.uniform(2, 5),
            'temperature_min': latest['temperature'] - np.random.uniform(3, 7),
            'humidity': latest['humidity'],
            'wind_speed': latest['wind_speed'],
            'wind_speed_10m': latest['wind_speed'] * 1.2,
            'rainfall_today': latest['rainfall'],
            'rainfall_7day': round(sum(h['rainfall'] for h in historical[-7:]), 1),
            'rainfall_30day': round(sum(h['rainfall'] for h in historical), 1),
            'temp_7day_avg': round(np.mean([h['temperature'] for h in historical[-7:]]), 1),
            'pressure': round(1013 + np.random.randn() * 10, 1),
            'specific_humidity': round(10 + np.random.randn() * 2, 2),
            'data_source': 'Simulated Data (API unavailable)',
            'date': datetime.now().strftime('%Y%m%d'),
            'province': NEPAL_DISTRICTS.get(district_name, {}).get('province', 'Unknown'),
            'region': NEPAL_DISTRICTS.get(district_name, {}).get('region', 'Hilly'),
            'historical': historical,
            'hazards': [{'type': '✅ NORMAL', 'severity': 'LOW', 'info': 'No active hazards'}]
        }

# ============================================
# INTERACTIVE VISUALIZATIONS
# ============================================

def create_historical_trend_chart(historical_data):
    """Create interactive historical trend chart"""
    if not historical_data:
        return go.Figure()
    
    df = pd.DataFrame(historical_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['temperature'],
        mode='lines+markers',
        name='Temperature (°C)',
        line=dict(color='#ff6b6b', width=2),
        marker=dict(size=4),
        yaxis='y1'
    ))
    
    fig.add_trace(go.Bar(
        x=df['date'], y=df['rainfall'],
        name='Rainfall (mm)',
        marker_color='#4ecdc4',
        opacity=0.7,
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['humidity'],
        mode='lines',
        name='Humidity (%)',
        line=dict(color='#95e77e', width=2, dash='dot'),
        yaxis='y1'
    ))
    
    fig.update_layout(
        title='📊 30-Day Weather History',
        xaxis_title='Date',
        hovermode='x unified',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis=dict(title='Temperature / Humidity', side='left', range=[0, 100]),
        yaxis2=dict(title='Rainfall (mm)', side='right', overlaying='y', showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_risk_timeline(historical_data, risk_calculator):
    """Create risk timeline chart"""
    if not historical_data:
        return go.Figure()
    
    df = pd.DataFrame(historical_data)
    
    risks = []
    for _, row in df.iterrows():
        score = (row['rainfall'] * 0.6 + row['wind_speed'] * 0.3 + row['humidity'] * 0.1)
        risks.append(min(75, max(0, score)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=risks,
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#f43b47', width=3),
        marker=dict(size=6, color='#ff6b6b'),
        fill='tozeroy',
        fillcolor='rgba(244, 59, 71, 0.1)'
    ))
    
    fig.add_hrect(y0=0, y1=20, fillcolor="#84fab0", opacity=0.3, line_width=0)
    fig.add_hrect(y0=20, y1=40, fillcolor="#ffe259", opacity=0.3, line_width=0)
    fig.add_hrect(y0=40, y1=60, fillcolor="#ff9a9e", opacity=0.3, line_width=0)
    fig.add_hrect(y0=60, y1=75, fillcolor="#f43b47", opacity=0.3, line_width=0)
    
    fig.update_layout(
        title='⚠️ Risk Level Timeline (30 Days)',
        xaxis_title='Date',
        yaxis_title='Risk Score (0-75)',
        height=350,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_weather_radar(weather):
    """Create radar chart for current weather"""
    categories = ['Temperature', 'Rainfall', 'Wind', 'Humidity']
    
    temp_norm = min(100, max(0, (weather['temperature'] / 45) * 100))
    rain_norm = min(100, max(0, (weather['rainfall_today'] / 150) * 100))
    wind_norm = min(100, max(0, (weather['wind_speed'] / 100) * 100))
    humidity_norm = min(100, max(0, weather['humidity']))
    
    values = [temp_norm, rain_norm, wind_norm, humidity_norm]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker=dict(color='#667eea', size=8),
        line=dict(color='#764ba2', width=2),
        name='Current'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12))
        ),
        title='🌡️ Weather Impact Radar',
        height=350,
        showlegend=False
    )
    
    return fig

def create_rainfall_forecast(weather):
    """Create rainfall comparison chart"""
    fig = go.Figure(data=[
        go.Bar(name='Today', x=['Rainfall'], y=[max(0, weather['rainfall_today'])], marker_color='#4ecdc4'),
        go.Bar(name='7-Day Avg', x=['Rainfall'], y=[max(0, weather['rainfall_7day'] / 7)], marker_color='#95e77e'),
        go.Bar(name='30-Day Avg', x=['Rainfall'], y=[max(0, weather['rainfall_30day'] / 30)], marker_color='#ffe259')
    ])
    
    fig.update_layout(
        title='📈 Rainfall Comparison (Daily Average)',
        yaxis_title='Rainfall (mm)',
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ============================================
# MODEL PREDICTOR
# ============================================

class DisasterRiskPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = None
        try:
            self.model = joblib.load(model_path)
        except:
            self.model = None
    
    def predict_risk_score(self, district, weather_data):
        """Calculate risk score based on weather data"""
        rainfall = max(0, weather_data.get('rainfall_today', 0))
        wind = max(0, weather_data.get('wind_speed', 0))
        humidity = min(100, max(0, weather_data.get('humidity', 60)))
        
        # Weighted risk calculation
        risk = (rainfall * 0.6) + (wind * 0.25) + (humidity * 0.15)
        
        # Region adjustment
        region = weather_data.get('region', 'Hilly')
        if region in ['Hilly', 'Mountain']:
            risk = risk * 1.2  # Higher risk in hilly areas for landslides
        
        # Rainfall pattern adjustment
        rain_7day = max(0, weather_data.get('rainfall_7day', 0))
        if rain_7day > 100:
            risk = risk * 1.3  # Saturated soil increases risk
        
        return np.clip(risk, 0, 75)
    

# ============================================
# AI ADVISOR
# ============================================

class NepalAIAdvisor:
    def __init__(self, gemini_model):
        self.model = gemini_model
    
    def get_advice(self, district, risk_score, risk_info, weather_data):
        # If Gemini is not available or failed, use fallback
        if self.model is None:
            return self._get_fallback_info(district, weather_data, risk_info)
        
        hazard_list = "\n".join([f"- {h['type']}: {h['info']}" for h in weather_data.get('hazards', [])])
        
        prompt = f"""
        Create an engaging, informative weather and disaster report for {district}, Nepal.
        
        Current Conditions:
        • Temperature: {weather_data.get('temperature', 'N/A')}°C
        • Humidity: {weather_data.get('humidity', 'N/A')}%
        • Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h
        • Rainfall Today: {weather_data.get('rainfall_today', 'N/A')} mm
        • Rainfall this week: {weather_data.get('rainfall_7day', 'N/A')} mm
        
        Active Hazards:
        {hazard_list if hazard_list else '- No active hazards'}
        
        Risk Level: {risk_info['level']} (Score: {risk_score:.0f}/75)
        
        Write a user-friendly report with these sections:
        
        🌤️ **Current Weather Summary** (1-2 sentences)
        
        ⚠️ **What You Should Know** (3 bullet points about current conditions)
        
        💡 **Practical Tips** (3 actionable tips for today)
        
        📢 **Stay Connected** (local resources and updates)
        
        Make it conversational and helpful, not alarming. Keep it concise (max 200 words).
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._get_fallback_info(district, weather_data, risk_info)
    
    def _get_fallback_info(self, district, weather_data, risk_info):
        temp = weather_data.get('temperature', 'N/A')
        humidity = weather_data.get('humidity', 'N/A')
        rainfall = weather_data.get('rainfall_today', 0)
        wind = weather_data.get('wind_speed', 0)
        
        # Ensure values are valid
        if isinstance(temp, (int, float)) and abs(temp) > 100:
            temp = 'N/A'
        if isinstance(humidity, (int, float)) and (humidity < 0 or humidity > 100):
            humidity = 'N/A'
        
        # Return clean text without HTML markup
        return f"""
**Current Weather Summary**

The weather in {district} currently shows {temp if temp != 'N/A' else 'moderate'}°C with {humidity if humidity != 'N/A' else 'moderate'}% humidity.

**What You Should Know**
• {rainfall:.1f}mm of rainfall recorded today
• Wind speeds at {wind:.1f} km/h
• {risk_info['level']} risk level for the area

**Practical Tips**
• Stay updated with local weather forecasts
• Keep emergency contacts handy: 1099
• Avoid unnecessary travel if conditions worsen

**For emergencies in Nepal, call 1099**
"""
# ============================================
# UI COMPONENTS
# ============================================

def get_risk_card(risk_score, risk_info):
    risk_class = {'LOW': 'risk-low', 'MODERATE': 'risk-moderate', 
                  'HIGH': 'risk-high', 'SEVERE': 'risk-severe'}.get(risk_info['level'], 'risk-moderate')
    
    return f"""
    <div class="{risk_class}">
        <h3 style="margin:0; opacity:0.8;">Current Risk Level</h3>
        <h1 style="margin:0.2rem 0; font-size:3rem;">{risk_info['level']}</h1>
        <p style="margin:0; font-size:1.1rem;">{risk_info['action']}</p>
        <p style="margin:0.5rem 0 0 0; font-size:2rem; font-weight:bold;">{risk_score:.0f}<span style="font-size:1rem;">/75</span></p>
    </div>
    """

def display_hazards(hazards):
    for hazard in hazards:
        severity_color = {"HIGH": "#f43b47", "MODERATE": "#ffa751", "LOW": "#84fab0"}.get(hazard['severity'], "#667eea")
        st.markdown(f"""
        <div class="hazard-card" style="border-left-color: {severity_color};">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">{hazard['type'].split()[0]}</span>
                <div style="flex: 1;">
                    <strong style="font-size: 1rem;">{hazard['type']}</strong>
                    <p style="margin: 0.2rem 0 0 0; color: #666;">{hazard['info']}</p>
                </div>
                <span style="font-size: 0.8rem; padding: 0.2rem 0.6rem; background: {severity_color}20; border-radius: 20px; color: {severity_color};">
                    {hazard['severity']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_risk_gauge(risk_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Risk Score", "font": {"size": 16}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 75], 'tickwidth': 1},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#84fab0'},
                {'range': [20, 40], 'color': '#ffe259'},
                {'range': [40, 60], 'color': '#ff9a9e'},
                {'range': [60, 75], 'color': '#f43b47'}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;">🌊 Nepal Disaster Risk Assessment System</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.9;">Real-time NASA satellite data • All 77 districts • Interactive visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize
    @st.cache_resource
    def init_system():
        return NASAPowerAPI(), DisasterRiskPredictor(), RiskAnalyzer(), NepalAIAdvisor(gemini_model)
    
    class RiskAnalyzer:
        def get_risk_level(self, score):
            if score < 20: return {'level': 'LOW', 'action': 'Normal conditions'}
            elif score < 40: return {'level': 'MODERATE', 'action': 'Be cautious'}
            elif score < 60: return {'level': 'HIGH', 'action': 'Prepare for emergency'}
            else: return {'level': 'SEVERE', 'action': 'Take action now'}
    
    nasa_api, predictor, analyzer, advisor = init_system()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📍 Select Location")
        
        provinces = ['All'] + sorted(list(set([d['province'] for d in NEPAL_DISTRICTS.values()])))
        selected_province = st.selectbox("Filter by Province", provinces)
        
        search = st.text_input("🔍 Search district", placeholder="Type district name...")
        
        districts_list = list(NEPAL_DISTRICTS.keys())
        if selected_province != 'All':
            districts_list = [d for d in districts_list if NEPAL_DISTRICTS[d]['province'] == selected_province]
        if search:
            districts_list = [d for d in districts_list if search.lower() in d.lower()]
        
        district = st.selectbox("Choose district", districts_list, label_visibility="collapsed")
        
        if district in NEPAL_DISTRICTS:
            info = NEPAL_DISTRICTS[district]
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;">
                <small>📍 {info['province']}</small><br>
                <small>🏔️ {info['region']} Region</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("📡 Data: NASA POWER Satellite")
        st.caption("🕐 Updated: Daily")
        st.caption(f"🏔️ {len(districts_list)} districts available")
        
        analyze_btn = st.button("🚀 Analyze Risk", type="primary", use_container_width=True)
    
    # Main content
    if analyze_btn:
        with st.spinner(f"Fetching NASA satellite data for {district}..."):
            weather = nasa_api.get_current_weather(district)
            risk_score = predictor.predict_risk_score(district, weather)
            risk_info = analyzer.get_risk_level(risk_score)
            ai_response = advisor.get_advice(district, risk_score, risk_info, weather) if advisor else None
            
            st.session_state.results = {
                'district': district, 'weather': weather, 'risk_score': risk_score,
                'risk_info': risk_info, 'ai_response': ai_response, 'timestamp': datetime.now()
            }
    
    # Display results
    if 'results' in st.session_state:
        r = st.session_state.results
        weather = r['weather']
        
        # Row 1: Risk Overview
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(get_risk_card(r['risk_score'], r['risk_info']), unsafe_allow_html=True)
        with col2:
            st.plotly_chart(create_risk_gauge(r['risk_score']), use_container_width=True)
        
        # Row 2: Interactive Visualizations
        st.markdown("### 📊 Interactive Weather Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Historical Trends", "⚠️ Risk Timeline", "🌡️ Weather Radar", "📉 Rainfall Analysis"])
        
        with tab1:
            if weather.get('historical'):
                st.plotly_chart(create_historical_trend_chart(weather['historical']), use_container_width=True)
            else:
                st.info("Historical data loading...")
        
        with tab2:
            if weather.get('historical'):
                st.plotly_chart(create_risk_timeline(weather['historical'], None), use_container_width=True)
            else:
                st.info("Risk timeline loading...")
        
        with tab3:
            st.plotly_chart(create_weather_radar(weather), use_container_width=True)
        
        with tab4:
            st.plotly_chart(create_rainfall_forecast(weather), use_container_width=True)
        
        # Row 3: Rainfall & Hazards
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Rainfall Summary")
            st.markdown(f"""
            <div style="background: white; border-radius: 15px; padding: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                    <span>Today:</span>
                    <strong>{max(0, weather['rainfall_today'])} mm</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                    <span>Last 7 days:</span>
                    <strong>{max(0, weather['rainfall_7day'])} mm</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                    <span>Last 30 days:</span>
                    <strong>{max(0, weather['rainfall_30day'])} mm</strong>
                </div>
                <div style="background: #e0e0e0; border-radius: 10px; height: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: {min(100, max(0, weather['rainfall_today']))}%; height: 100%;"></div>
                </div>
                <p style="margin-top: 0.5rem; color: #666; font-size: 0.8rem;">
                    {'⚠️ Above average rainfall' if weather['rainfall_today'] > 25 else '✅ Normal rainfall pattern'}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ⚠️ Active Hazards")
            display_hazards(weather.get('hazards', []))
        
        # Row 4: AI Recommendations
        if r['ai_response']:
           st.markdown("### 🤖 AI-Powered Insights")
    
    # Display the AI response directly (it will be rendered as markdown)
           st.markdown(r['ai_response'])
    
    # Then add the footer with styling
        st.markdown("---")
        st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
        <small>📡 Based on NASA POWER satellite data</small>
        <small>🕐 Updated: {r['timestamp'].strftime('%H:%M:%S')}</small>
    </div>
    """, unsafe_allow_html=True)
        
if __name__ == "__main__":
    main()
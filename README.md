# 🌊 Nepal Disaster Risk Assessment System

An interactive, data-driven web portal engineered to analyze and forecast environmental and disaster threat vectors across all 77 districts of Nepal. The system ingests planetary remote sensing parameters dynamically from the **NASA POWER API** and couples a weighted climatological heuristic matrix with **Gemini 1.5 Flash LLM** telemetry to generate high-fidelity, real-time risk profiles.

---

## 🚀 Key Features

* **Complete 77 District Matrix:** Pre-mapped geographical, provincial, and topographical coordinates covering the entirety of Nepal.
* **Live Satellite Telemetry Ingestion:** Fetches real-time climate inputs (Daily Precipitation sums, Temperature profiles, Relative Humidity, Wind Vector velocities) straight from NASA's Power data endpoints.
* **Predictive Engineering Engine:** Computes immediate and cumulative risk scores (0-75) with dynamic penalty scaling rules for high-altitude terrains and sustained 7-day soil saturation variables.
* **Interactive Visualizations:** Renders comprehensive 30-day timelines, cross-vector weather radar graphs, and dual-axis historical trend lines powered by Plotly.
* **Dual-Layer Advisory Layer:** Pairs local conditional scripts with state-of-the-art LLM orchestration (Gemini API) to deliver zero-latency situational reports with automated fallback resilience.

---

## 🛠️ Architecture & System Topology

The platform coordinates real-time data ingestion, local analytics processing, and multi-threaded presentation layers seamlessly:

1. **Presentation (UI) Layer:** Streamlit runtime customized via explicit CSS formatting variables.
2. **Data Pipeline Layer:** A resilient caching subsystem featuring a 5-minute Time-To-Live (TTL) loop to handle API connection limits.
3. **Analytics Core:** A weighted modeling pipeline that combines current data windows with an option to connect pre-trained scikit-learn models (`joblib` serialization binary).

---

## 🏃 Getting Started

### 📋 Prerequisites
Ensure you have Python 3.9+ installed on your computer.

### 📦 Installation
Clone the repository and install the dependencies listed in the setup manifest:

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# Install required dependencies
pip install -r requirements.txt

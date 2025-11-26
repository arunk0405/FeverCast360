# FeverCast360 🌡️

## Intelligent Fever Outbreak Prediction & Management System

FeverCast360 is a comprehensive ML-powered dashboard for real-time fever outbreak prediction, monitoring, and resource management across Indian cities.

---

## 🌟 Key Features

### 🏛️ Government Dashboard
- **Interactive Map Visualization** - Real-time outbreak mapping with actual city boundaries
- **Risk Level Classification** - Critical/High/Moderate/Low severity indicators
- **Population Impact Analysis** - Automated risk assessment for affected populations
- **Priority Action Plans** - Automated response recommendations for high-risk districts

### 💊 Pharmaceutical Dashboard
- **Smart Stock Management** - Automated medicine demand forecasting
- **Regional Recommendations** - Targeted stock allocation based on severity
- **Real-time Dispatch** - One-click stock order dispatch
- **Demand Analytics** - Visual insights into medicine requirements

### 👥 Public Awareness Dashboard
- **City-wise Risk Search** - Instant risk status for any city
- **Health Advisories** - Contextual prevention and treatment guidelines
- **Risk Level Education** - Clear explanation of safety measures
- **Location-based Alerts** - Personalized health risk information

### ⚙️ ML Pipeline
- **Two-Stage Prediction Model**
  - Stage 1: Logistic Regression for outbreak probability
  - Stage 2: Random Forest/XGBoost for fever type classification
- **Automatic Geocoding** - Dynamic city location detection via OpenStreetMap
- **Smart Caching** - Efficient reuse of geocoded data
- **Firebase Integration** - Cloud-based prediction storage

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Firebase account with Firestore enabled
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd FeverML
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Firebase**
   - Place your Firebase service account key as `newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json` in the project root
   - See [FIREBASE_SETUP.md](FIREBASE_SETUP.md) for detailed instructions

5. **Run the application**
```bash
streamlit run app.py
```

6. **Access the dashboard**
   - Open your browser to `http://localhost:8501`

---

## 📊 Usage Guide

### Running ML Pipeline

1. Navigate to the **ML Pipeline** tab
2. Upload a preprocessed CSV with columns:
   - `Region` - City/district name
   - `Temperature` - Average temperature (°C)
   - `Humidity` - Humidity percentage
   - `Rainfall` - Rainfall in mm
   - `Population_Density` - People per sq km
   - `Mosquito_Index` - Mosquito prevalence (0-1)
   - `Sanitation_Score` - Sanitation rating (0-10)
   - `outbreak_label` - Historical outbreak indicator (0/1)
   - `fever_type` - Known fever type (if any)

3. Click **Run ML Pipeline & Save Results**
4. The system will:
   - Train prediction models
   - Generate predictions for all regions
   - Automatically geocode cities
   - Save data to Firebase
   - Display results with risk classification

### Viewing Predictions

- **Government View**: See all high-risk regions on an interactive map
- **Pharma View**: Check medicine stock requirements by region
- **Public View**: Search specific cities for risk status

---

## 🏗️ Project Structure

```
FeverML/
├── app.py                          # Main Streamlit application
├── db_utils.py                     # Firebase database operations
├── prediction.py                   # ML pipeline implementation
├── requirements.txt                # Python dependencies
├── fevercast360_sample_dataset.csv # Sample data for testing
├── newp-9a65c-firebase-...json    # Firebase credentials (not in git)
├── models/                         # Trained ML models (auto-generated)
├── outputs/                        # Prediction outputs (auto-generated)
├── README.md                       # This file
├── FIREBASE_SETUP.md              # Firebase configuration guide
└── MIGRATION_GUIDE.md             # Database migration documentation
```

---

## 🔧 Technical Stack

### Frontend
- **Streamlit** - Web dashboard framework
- **Folium** - Interactive map visualization
- **Plotly** - Data visualization charts

### Backend
- **Firebase Firestore** - NoSQL cloud database
- **OpenStreetMap Nominatim API** - Geocoding service

### Machine Learning
- **scikit-learn** - Logistic Regression, Random Forest
- **XGBoost** (optional) - Advanced gradient boosting
- **pandas/numpy** - Data processing

---

## 📋 Database Schema

### Collections in Firestore

#### `predictions`
```javascript
{
  "region": "string",
  "p_outbreak": "float",      // Outbreak probability (0-1)
  "fever_type": "string",      // Dengue/Typhoid/Viral/None
  "p_type": "float",           // Classification confidence
  "severity_index": "float",   // Combined risk score (0-1)
  "ts": "timestamp"
}
```

#### `region_metadata`
```javascript
{
  "lat": "float",
  "lon": "float",
  "population": "integer",
  "state": "string"
}
```

#### `pharma_stock`
```javascript
{
  "paracetamol": "integer",
  "ors": "integer",
  "antibiotics": "integer",
  "iv_fluids": "integer",
  "last_updated": "timestamp"
}
```

---

## 🎨 Features in Detail

### Dynamic Geocoding
- Automatically fetches latitude/longitude for any Indian city
- Multiple search strategies for accuracy
- Smart caching to avoid redundant API calls
- Handles rate limiting gracefully

### Interactive Map
- Actual city boundary polygons from OpenStreetMap
- Color-coded risk levels (Green/Amber/Orange/Red)
- Dynamic opacity based on severity
- Hover tooltips and detailed popups
- Zoom/pan controls for easy navigation

### Smart Stock Allocation
- Population-based demand calculation
- Severity-weighted distribution
- Region-specific medicine requirements
- One-click dispatch system

---

## 🔐 Security Notes

- **Never commit** Firebase service account keys to version control
- Add `*.json` to `.gitignore` (except sample files)
- Use environment variables for sensitive configuration
- Enable Firebase security rules in production

---

## 🐛 Troubleshooting

### Firebase API Not Enabled
```
Error: Cloud Firestore API has not been used
```
**Solution**: Visit the Firebase Console and enable Cloud Firestore API
- URL: https://console.developers.google.com/apis/api/firestore.googleapis.com/

### Geocoding Rate Limit
```
Error: 429 Too Many Requests
```
**Solution**: The system automatically handles rate limiting with delays. If persistent, reduce the number of regions processed simultaneously.

### Map Not Showing Cities
```
No geocoded regions available
```
**Solution**: 
1. Check Firebase connectivity
2. Verify regions have lat/lon in `region_metadata` collection
3. Use the "Re-Geocode Regions Without Coordinates" button in ML Pipeline tab

---

## 📈 Performance Tips

1. **Reuse Geocoded Data**: Once geocoded, city coordinates are cached in Firebase
2. **Batch Processing**: Process multiple regions in one ML pipeline run
3. **Connection Pooling**: Firebase handles connection management automatically
4. **Map Loading**: City boundaries are fetched progressively with a progress bar

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- FeverCast360 Team
- Contact: [Your contact information]

---

## 🙏 Acknowledgments

- OpenStreetMap for geocoding services
- Firebase for cloud infrastructure
- Streamlit community for excellent documentation
- Folium contributors for mapping capabilities

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check [FIREBASE_SETUP.md](FIREBASE_SETUP.md) for setup help
- Review [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for database info

---

**Built with ❤️ for public health monitoring**

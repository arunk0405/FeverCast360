# FeverCast360 Enhanced ML Pipeline

## 🎯 Overview

The Enhanced ML Pipeline uses **historical disease data** combined with environmental and infrastructure factors to predict future outbreak risks. This is a significant improvement over the legacy pipeline as it learns from actual disease patterns.

---

## 🔄 How It Works

### Stage 1: Training (Historical Data Learning)
The model learns relationships between:
- **Environmental Factors**: Temperature, Humidity, Rainfall
- **Infrastructure**: Sanitation Score, Population Density
- **Disease Outcomes**: Dengue, Malaria, Chikungunya cases

**Output**: A trained model that understands how environmental conditions lead to disease outbreaks

### Stage 2: Prediction (Future Outbreak Forecasting)
Using only current environmental data (no disease cases needed), the model predicts:
- **Severity Index**: Expected disease burden (0-1 scale)
- **Outbreak Probability**: Likelihood of outbreak (0-1 scale)
- **Risk Category**: Low / Moderate / High

---

## 📊 Data Requirements

### Training Data (CSV with these columns):

| Column | Description | Example |
|--------|-------------|---------|
| `State` | State name | Karnataka |
| `District` | District/City name | Bengaluru |
| `Year` | Year of data | 2022 |
| `Month` | Month (1-12) | 6 |
| `Dengue_Cases` | Number of dengue cases | 280 |
| `Malaria_Cases` | Number of malaria cases | 60 |
| `Chikungunya_Cases` | Number of chikungunya cases | 40 |
| `Temperature` | Avg temperature (°C) | 26.5 |
| `Humidity` | Avg humidity (%) | 80 |
| `Rainfall` | Total rainfall (mm) | 125.3 |
| `Sanitation_Score` | Sanitation index (0-1) | 0.85 |
| `Population_Density` | People per km² | 4000 |

**Sample**: See `sample_training_data.csv`

### Prediction Data (CSV with these columns):

| Column | Description | Example |
|--------|-------------|---------|
| `State` | State name | Karnataka |
| `District` | District/City name | Bengaluru |
| `Year` | Year to predict | 2025 |
| `Month` | Month (1-12) | 1 |
| `Temperature` | Expected temperature (°C) | 26.0 |
| `Humidity` | Expected humidity (%) | 68 |
| `Rainfall` | Expected rainfall (mm) | 10.0 |
| `Sanitation_Score` | Current sanitation (0-1) | 0.87 |
| `Population_Density` | People per km² | 4200 |

**Sample**: See `sample_prediction_data.csv`

---

## 🚀 Usage

### Option 1: Via Streamlit App (Recommended)

1. **Open ML Pipeline** in sidebar
2. **Select "Enhanced Pipeline"** mode
3. **Training Tab**:
   - Upload training CSV with historical disease data
   - Click "Train Enhanced Model"
   - Wait for training to complete (2-5 minutes)
4. **Prediction Tab**:
   - Upload prediction CSV with environmental data only
   - Click "Generate Predictions & Save to Firebase"
   - View results and download predictions

### Option 2: Command Line

**Training:**
```bash
python prediction_enhanced.py \
  --mode train \
  --training_data sample_training_data.csv \
  --models_dir models_enhanced
```

**Prediction:**
```bash
python prediction_enhanced.py \
  --mode predict \
  --prediction_data sample_prediction_data.csv \
  --models_dir models_enhanced \
  --output outputs/predictions_enhanced.csv
```

### Option 3: Python Script

```python
import prediction_enhanced

# Training
predictor = prediction_enhanced.run_enhanced_pipeline_train(
    training_csv="sample_training_data.csv",
    models_dir="models_enhanced"
)

# Prediction
results = prediction_enhanced.run_enhanced_pipeline_predict(
    prediction_csv="sample_prediction_data.csv",
    models_dir="models_enhanced"
)

print(results)
```

---

## 📈 Model Architecture

### Severity Index Calculation
```
Severity = (Dengue × 0.4 + Malaria × 0.35 + Chikungunya × 0.25) / Population_Density × 100,000
```
Then normalized to 0-1 scale.

### Machine Learning Models

1. **Random Forest Regressor** (Severity Prediction)
   - 200 trees
   - Max depth: 15
   - Predicts continuous severity index

2. **Gradient Boosting Classifier** (Outbreak Classification)
   - 150 estimators
   - Learning rate: 0.1
   - Binary classification (Outbreak / No Outbreak)

### Feature Engineering

**Temporal Features:**
- `Month_Sin` = sin(2π × Month / 12)
- `Month_Cos` = cos(2π × Month / 12)

**Interaction Features:**
- `Temp_Humidity` = Temperature × Humidity / 100
- `Rain_Sanitation` = Rainfall × (1 - Sanitation_Score)
- `Temp_Rainfall` = Temperature × Rainfall

---

## 📊 Output Format

Predictions are saved with these columns:

| Column | Description | Range |
|--------|-------------|-------|
| `State` | State name | - |
| `District` | District name | - |
| `Year` | Prediction year | - |
| `Month` | Prediction month | 1-12 |
| `Predicted_Severity_Index` | Expected severity | 0-1 |
| `Outbreak_Probability` | Outbreak likelihood | 0-1 |
| `Outbreak_Risk` | Risk category | Low/Moderate/High |
| `Temperature` | Input temperature | °C |
| `Humidity` | Input humidity | % |
| `Rainfall` | Input rainfall | mm |
| `Sanitation_Score` | Input sanitation | 0-1 |
| `Population_Density` | Input density | per km² |

---

## 🎯 Key Advantages Over Legacy Pipeline

| Feature | Legacy Pipeline | Enhanced Pipeline |
|---------|----------------|-------------------|
| Uses historical disease data | ❌ No | ✅ Yes |
| Learns disease patterns | ❌ Limited | ✅ Advanced |
| Severity calculation | Basic features only | Disease cases + environment |
| Prediction accuracy | Moderate | High |
| Training required | No separate training | Yes (one-time) |
| Interpretability | Limited | Feature importance available |

---

## 🔧 Model Files

After training, these files are saved in `models_enhanced/`:

- `severity_model.pkl` - Random Forest regressor
- `outbreak_classifier.pkl` - Gradient Boosting classifier
- `scaler.pkl` - Feature scaler
- `state_encoder.pkl` - State label encoder
- `district_encoder.pkl` - District label encoder
- `feature_cols.txt` - Feature column names

**Total size**: ~5-10 MB

---

## 💡 Tips for Best Results

1. **Training Data Quality**:
   - Use at least 50+ rows for training
   - Include data from multiple seasons/months
   - Cover diverse geographical regions
   - Ensure accurate disease case counts

2. **Prediction Accuracy**:
   - Use recent meteorological forecasts
   - Update sanitation scores periodically
   - Retrain model annually with new data

3. **Handling New Districts**:
   - Model can predict for new districts not in training data
   - However, accuracy is best for known districts
   - Include similar districts in training for better generalization

4. **Seasonal Patterns**:
   - Model automatically captures seasonal variations via cyclical month encoding
   - Include full year of data in training for best results

---

## 🔬 Model Performance Metrics

After training, you'll see:

- **R² Score** (Severity Model): Measures prediction accuracy (closer to 1 is better)
- **MSE/RMSE**: Mean squared error (lower is better)
- **Classification Accuracy**: Outbreak prediction accuracy (higher is better)
- **Feature Importance**: Which factors matter most

---

## 🆚 When to Use Which Pipeline?

**Use Enhanced Pipeline When:**
- ✅ You have historical disease case data
- ✅ You want accurate outbreak predictions
- ✅ You need severity index calculations
- ✅ You're doing long-term planning

**Use Legacy Pipeline When:**
- ❌ No historical disease data available
- ❌ Quick baseline prediction needed
- ❌ Exploring different feature sets
- ❌ Testing new data formats

---

## 📚 References

- **Random Forest**: Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- **Gradient Boosting**: Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
- **Disease Prediction**: WHO guidelines on epidemic forecasting

---

## 🐛 Troubleshooting

### "Missing required columns" error
**Solution**: Ensure your CSV has all required columns with exact names (case-sensitive)

### "No trained models found" error
**Solution**: Train the model first in the Training tab before making predictions

### Low prediction accuracy
**Solutions**:
- Add more diverse training data
- Include more historical months
- Verify data quality (no outliers or errors)
- Ensure consistent units (temperature in °C, rainfall in mm)

### Out of memory error
**Solution**: Reduce training data size or use a machine with more RAM (models need ~2GB for large datasets)

---

**Last Updated**: November 29, 2025  
**Version**: 1.0.0  
**Author**: FeverCast360 Team

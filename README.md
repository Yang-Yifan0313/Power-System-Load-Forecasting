# Power System Load Forecasting

## 📋 Project Overview
This project implements an ensemble learning approach for power load forecasting, combining Random Forest and LightGBM models to capture both short-term dependencies and long-term patterns in power consumption data. The model achieves high accuracy in predicting power load values by leveraging temporal features and environmental factors.

## 🌟 Key Features
- Ensemble model combining Random Forest (60%) and LightGBM (40%)
- Comprehensive feature engineering including temporal, cyclical, and meteorological features
- Strong capability in capturing both short-term and long-term power load patterns
- Advanced parameter tuning based on multiple evaluation metrics

## 🛠 Technical Stack
- **Primary Models**: 
  - Random Forest
  - LightGBM
- **Key Libraries**:
  - scikit-learn
  - lightgbm
  - pandas
  - numpy
  - matplotlib/seaborn (for visualization)

## 🔧 Feature Engineering
The project implements comprehensive feature engineering including:

1. **Temporal Lag Features**
   - Historical load values from previous 1-5 days
   - Same-hour loads from previous day/week

2. **Cyclical Time Features**
   - Trigonometric encoding of time periodicity
   - Hourly, daily, weekly, and monthly patterns

3. **Environmental and Special Features**
   - Holiday indicators
   - Meteorological data (temperature, humidity, wind speed, precipitation)
   - Time period markers (peak hours, night periods)

## 📊 Model Performance
The ensemble model demonstrates excellent performance in prediction tasks:
- Close alignment between predicted and actual values
- Strong capture of cyclical patterns
- Effective handling of both short-term dependencies and long-term trends

### Feature Importance
- Random Forest emphasizes temporal continuity
- LightGBM shows balanced importance across various feature types
- Complementary strengths in capturing different aspects of power load patterns

## 📈 Evaluation Metrics
The model is evaluated using multiple metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

## 🚀 Getting Started

### Prerequisites
- Python 3.9 (required)
  - Note: This project is specifically developed and tested with Python 3.9
  - Other versions may cause compatibility issues
- pip (Python package manager)

### Required Libraries
```bash
numpy
pandas
scikit-learn
lightgbm
holidays
matplotlib
seaborn
math
```

### Installation
1. Clone this repository
```bash
git clone https://github.com/Yang-Yifan0313/Power-System-Load-Forecasting.git
cd Power-System-Load-Forecasting
```


### Usage
### Dataset Structure

The project includes the following data files:

1. **Training Data**: `Train data.csv`
   - Location: Root directory
   - Sampling Frequency: Hourly (24 records per day)
   - Required fields:
     - DateTime: Timestamp for each record (hourly intervals)
     - Load: Power load value
     - Temperature: Environmental temperature
     - Humidity: Environmental humidity percentage
     - Wind_speed: Wind speed measurement
     - Precipitation: Precipitation measurement

Example of data format:
```csv
DateTime           Precipitation Temperature Wind_speed Humidity    Load
2020/1/1 0:00     0            3.815      5.565      74.9025     436.4835
2020/1/1 1:00     0            3.7775     5.495      75.175      426.8823
2020/1/1 2:00     0            3.7675     5.625      76.34       403.6138
2020/1/1 3:00     0            3.8025     5.285      78.255      384.7519
```

2. **Test Data**: Located in `Test` folder
   - Monthly test files (same hourly sampling frequency):
     - February.csv
     - April.csv
     - June.csv
     - August.csv
     - October.csv
     - December.csv
   - Answer.csv (for predictions output)

Note: The time series data is continuous with one record per hour. Each day contains 24 records, which is crucial for the feature engineering process, especially for creating lag features and capturing daily patterns.

The data files are already included in the repository and contain all necessary fields for the model to work.

2. Run the main script:
```bash
python main.py
```

3. Check the results:
   - Model evaluation results will be saved in 'model_evaluation_results.csv'
   - Visualizations will be saved in the current directory:
     - actual_vs_predicted.png
     - error_distribution.png
     - model_comparison.png
     - time_patterns.png
     - residual_plot.png
     - correlation_matrix.png
     - rf_feature_importance.png
     - lgb_feature_importance.png
     - seasonal_patterns.png
   - Final predictions will be saved in 'Test/Answer.csv'

## 📝 Results
-Ranking: 30th place out of 288 participants in the course Kaggle competition

The model was evaluated in a Kaggle competition specifically organized for the ELEC7011 course. The final score and ranking reflect the model's performance among 288 course participants, placing in the top 10.4% of all submissions.

## 📊 Visualizations
The project includes three key visualizations:
1. Actual vs Predicted Load Values
2. Feature Correlation Matrix
3. Feature Importance Distributions for both Random Forest and LightGBM

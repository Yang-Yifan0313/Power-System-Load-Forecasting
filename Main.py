import holidays
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def create_performance_visualizations(train_data, y_true, y_pred_rf, y_pred_lgb, y_pred_ensemble, results_df):
    """
    创建模型性能相关的可视化图表

    Parameters:
        train_data: 训练数据DataFrame
        y_true: 实际值
        y_pred_rf: Random Forest预测值
        y_pred_lgb: LightGBM预测值
        y_pred_ensemble: 集成模型预测值
        results_df: 包含模型评估结果的DataFrame
    """
    # 设置图表风格
    plt.style.use('seaborn')

    # 1. 预测值vs实际值对比图
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred_ensemble, label='Ensemble Prediction', alpha=0.7)
    plt.title('Actual vs Predicted Load Values')
    plt.xlabel('Time Points')
    plt.ylabel('Load')
    plt.legend()
    plt.savefig('actual_vs_predicted.png')
    plt.close()

    # 2. 预测误差分布图
    plt.figure(figsize=(12, 6))
    errors_rf = y_pred_rf - y_true
    errors_lgb = y_pred_lgb - y_true
    errors_ensemble = y_pred_ensemble - y_true

    plt.hist(errors_rf, bins=50, alpha=0.5, label='Random Forest', density=True)
    plt.hist(errors_lgb, bins=50, alpha=0.5, label='LightGBM', density=True)
    plt.hist(errors_ensemble, bins=50, alpha=0.5, label='Ensemble', density=True)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('error_distribution.png')
    plt.close()

    # 3. 模型性能指标比较图
    plt.figure(figsize=(12, 6))
    metrics = ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE']
    rf_metrics = results_df[results_df['Model'] == 'Random Forest'][metrics].mean()
    lgb_metrics = results_df[results_df['Model'] == 'LightGBM'][metrics].mean()

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width / 2, rf_metrics, width, label='Random Forest')
    plt.bar(x + width / 2, lgb_metrics, width, label='LightGBM')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.savefig('model_comparison.png')
    plt.close()

    # 4. 时间特征重要性热力图
    plt.figure(figsize=(15, 8))
    hour_importance = pd.DataFrame({
        'Hour': range(24),
        'Load': train_data.groupby(train_data['DateTime'].dt.hour)['Load'].mean()
    })

    pivot_data = train_data.pivot_table(
        values='Load',
        index=train_data['DateTime'].dt.hour,
        columns=train_data['DateTime'].dt.dayofweek,
        aggfunc='mean'
    )

    sns.heatmap(pivot_data, cmap='YlOrRd', center=pivot_data.mean().mean())
    plt.title('Average Load by Hour and Day of Week')
    plt.xlabel('Day of Week (0=Monday)')
    plt.ylabel('Hour of Day')
    plt.savefig('time_patterns.png')
    plt.close()

    # 5. 模型预测残差图
    plt.figure(figsize=(15, 6))
    plt.scatter(y_pred_ensemble, errors_ensemble, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig('residual_plot.png')
    plt.close()

    # 6. 特征相关性热力图
    plt.figure(figsize=(12, 10))
    correlation_matrix = train_data.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()


def visualize_feature_importance(rf_model, lgb_model, feature_names):
    """
    创建特征重要性可视化

    Parameters:
        rf_model: 训练好的Random Forest模型
        lgb_model: 训练好的LightGBM模型
        feature_names: 特征名称列表
    """
    # Random Forest特征重要性
    plt.figure(figsize=(12, 6))
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.barh(range(len(rf_importance)), rf_importance['importance'])
    plt.yticks(range(len(rf_importance)), rf_importance['feature'])
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    plt.close()

    # LightGBM特征重要性
    plt.figure(figsize=(12, 6))
    lgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.barh(range(len(lgb_importance)), lgb_importance['importance'])
    plt.yticks(range(len(lgb_importance)), lgb_importance['feature'])
    plt.title('LightGBM Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('lgb_feature_importance.png')
    plt.close()


def plot_seasonal_patterns(train_data):
    """
    创建季节性模式的可视化

    Parameters:
        train_data: 训练数据DataFrame
    """
    # 按月份和小时的平均负载
    plt.figure(figsize=(15, 6))
    monthly_hourly_load = train_data.pivot_table(
        values='Load',
        index=train_data['DateTime'].dt.month,
        columns=train_data['DateTime'].dt.hour,
        aggfunc='mean'
    )

    sns.heatmap(monthly_hourly_load, cmap='YlOrRd')
    plt.title('Average Load by Month and Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Month')
    plt.savefig('seasonal_patterns.png')
    plt.close()

    # 按季节的负载箱型图
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Season', y='Load', data=train_data)
    plt.title('Load Distribution by Season')
    plt.savefig('seasonal_boxplot.png')
    plt.close()


def get_cyclical_features(value, max_value):
    """
    将周期性特征转换为正弦和余弦编码
    """
    value = value * 2 * math.pi / max_value
    return np.sin(value), np.cos(value)


def is_special_holiday(date, cn_holidays):
    """
    判断是否为特殊节假日（春节、国庆等主要节假日）
    """
    if date in cn_holidays:
        holiday_name = cn_holidays.get(date)
        major_holidays = ['春节', '元宵节', '清明节', '劳动节', '端午节', '中秋节', '国庆节']
        return any(holiday in holiday_name for holiday in major_holidays)
    return False


def preprocess_data(data, is_train=True):
    print(f"Input data shape: {data.shape}")
    data['DateTime'] = pd.to_datetime(data['DateTime'])

    # 1. 基础时间特征的周期性编码
    # 小时编码 (0-23)
    sin_hour, cos_hour = get_cyclical_features(data['DateTime'].dt.hour, 24)
    data['Hour_sin'] = sin_hour
    data['Hour_cos'] = cos_hour

    # 日期编码 (1-31)
    sin_day, cos_day = get_cyclical_features(data['DateTime'].dt.day, 31)
    data['Day_sin'] = sin_day
    data['Day_cos'] = cos_day

    # 月份编码 (1-12)
    sin_month, cos_month = get_cyclical_features(data['DateTime'].dt.month, 12)
    data['Month_sin'] = sin_month
    data['Month_cos'] = cos_month

    # 星期编码 (0-6)
    sin_week, cos_week = get_cyclical_features(data['DateTime'].dt.dayofweek, 7)
    data['Week_sin'] = sin_week
    data['Week_cos'] = cos_week

    # 2. 节假日特征
    cn_holidays = holidays.CN()  # 中国节假日

    # 添加节假日特征
    data['Is_holiday'] = data['DateTime'].map(lambda x: x in cn_holidays).astype(int)
    data['Is_special_holiday'] = data['DateTime'].map(lambda x: is_special_holiday(x, cn_holidays)).astype(int)

    # 添加节假日前后工作日特征
    data['Is_holiday_next_day'] = data['DateTime'].map(
        lambda x: (x + pd.Timedelta(days=1)) in cn_holidays).astype(int)
    data['Is_holiday_prev_day'] = data['DateTime'].map(
        lambda x: (x - pd.Timedelta(days=1)) in cn_holidays).astype(int)

    # 3. 其他时间特征
    data['Year'] = data['DateTime'].dt.year
    data['Is_workday'] = ((data['DateTime'].dt.dayofweek < 5) &
                          ~data['Is_holiday']).astype(int)
    data['Season'] = data['DateTime'].dt.month.map(
        lambda x: (x % 12 + 3) // 3)
    data['DayOfYear'] = data['DateTime'].dt.dayofyear
    data['WeekOfYear'] = data['DateTime'].dt.isocalendar().week
    data['IsWeekend'] = (data['DateTime'].dt.dayofweek >= 5).astype(int)

    # 4. 时间段特征
    data['Is_morning_peak'] = ((data['DateTime'].dt.hour >= 7) &
                               (data['DateTime'].dt.hour <= 9)).astype(int)
    data['Is_evening_peak'] = ((data['DateTime'].dt.hour >= 17) &
                               (data['DateTime'].dt.hour <= 19)).astype(int)
    data['Is_night'] = ((data['DateTime'].dt.hour >= 22) |
                        (data['DateTime'].dt.hour <= 5)).astype(int)

    if is_train:
        # 对于训练数据，添加滞后特征
        data['Load_prev_day'] = data['Load'].shift(24)
        data['Load_prev_week'] = data['Load'].shift(168)
        data['Load_prev_2days'] = data['Load'].shift(48)
        data['Load_prev_3days'] = data['Load'].shift(72)
        data['Load_prev_4days'] = data['Load'].shift(96)
        data['Load_prev_5days'] = data['Load'].shift(120)

        # 添加同期历史负载特征
        data['Load_same_hour_yesterday'] = data.groupby(data['DateTime'].dt.hour)['Load'].shift(24)
        data['Load_same_hour_lastweek'] = data.groupby(data['DateTime'].dt.hour)['Load'].shift(168)
    else:
        # 对于测试数据，谨慎处理滞后特征
        data['Load_prev_day'] = data['Load'].shift(24)
        data['Load_prev_week'] = data['Load'].shift(168)
        data['Load_prev_2days'] = data['Load'].shift(48)
        data['Load_prev_3days'] = data['Load'].shift(72)
        data['Load_prev_4days'] = data['Load'].shift(96)
        data['Load_prev_5days'] = data['Load'].shift(120)
        data['Load_same_hour_yesterday'] = data.groupby(data['DateTime'].dt.hour)['Load'].shift(24)
        data['Load_same_hour_lastweek'] = data.groupby(data['DateTime'].dt.hour)['Load'].shift(168)

        # 获取最后已知数据模式
        last_known_data = data.loc[data['Load'] != 0, 'Load']

        # 填充滞后特征中的NaN值
        for i in range(24):
            if data['Load'].iloc[-24 + i] == 0:
                data.loc[data.index[-24 + i], 'Load_prev_day'] = last_known_data.iloc[-24 + i]
                data.loc[data.index[-24 + i], 'Load_prev_week'] = last_known_data.iloc[-144 + i]
                data.loc[data.index[-24 + i], 'Load_prev_2days'] = last_known_data.iloc[-48 + i]
                data.loc[data.index[-24 + i], 'Load_prev_3days'] = last_known_data.iloc[-72 + i]
                data.loc[data.index[-24 + i], 'Load_prev_4days'] = last_known_data.iloc[-96 + i]
                data.loc[data.index[-24 + i], 'Load_prev_5days'] = last_known_data.iloc[-120 + i]
                data.loc[data.index[-24 + i], 'Load_same_hour_yesterday'] = last_known_data.iloc[-24 + i]
                data.loc[data.index[-24 + i], 'Load_same_hour_lastweek'] = last_known_data.iloc[-144 + i]

    return data


def process_features(data, feature_columns):
    return data[feature_columns].values


def calculate_metrics(y_true, y_pred):
    """
    计算多个评估指标
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # 计算MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def evaluate_with_time_series_cv(model, X, y, n_splits=5):
    """
    使用时间序列交叉验证评估模型
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        # 分割数据
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 训练模型
        model.fit(X_train, y_train)

        # 预测和评估
        y_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val, y_pred)
        metrics['fold'] = fold
        all_metrics.append(metrics)

        print(f"\nFold {fold} Results:")
        for metric_name, value in metrics.items():
            if metric_name != 'fold':
                print(f"{metric_name}: {value:.4f}")

    # 计算平均指标
    avg_metrics = {}
    std_metrics = {}
    for metric in ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE']:
        values = [m[metric] for m in all_metrics]
        avg_metrics[metric] = np.mean(values)
        std_metrics[metric] = np.std(values)

    return avg_metrics, std_metrics, all_metrics


def main():
    # 加载和预处理训练数据
    train_data = pd.read_csv('Train data.csv')
    train_data = preprocess_data(train_data, is_train=True)

    # 特征列表保持不变
    feature_columns = [
        'Load_prev_day', 'Load_prev_week', 'Load_prev_2days', 'Load_prev_3days',
        'Load_prev_4days', 'Load_prev_5days', 'Load_same_hour_yesterday',
        'Load_same_hour_lastweek',
        'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
        'Week_sin', 'Week_cos',
        'Is_holiday', 'Is_special_holiday', 'Is_holiday_next_day',
        'Is_holiday_prev_day',
        'Is_workday', 'Season', 'DayOfYear', 'WeekOfYear', 'IsWeekend',
        'Is_morning_peak', 'Is_evening_peak', 'Is_night',
        'Temperature', 'Humidity', 'Wind_speed', 'Precipitation'
    ]

    # 数据清理和准备
    train_data_cleaned = train_data.dropna().iloc[168:]
    X = process_features(train_data_cleaned, feature_columns)
    y = train_data_cleaned['Load'].values

    # 标准化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # 初始化模型
    rf_model = RandomForestRegressor(
        n_estimators=200,  # 增加树的数量,原来是100
        max_depth=15,  # 限制树的深度,避免过拟合
        min_samples_split=5,  # 调整分裂所需的最小样本数
        min_samples_leaf=2,  # 调整叶节点最小样本数
        max_features='sqrt',  # 使用特征的平方根数量
        random_state=42,
        n_jobs=-1  # 使用所有CPU核心以加速训练
    )
    lgb_model = LGBMRegressor(n_estimators=100, random_state=42)

    print("\nEvaluating Random Forest Model...")
    rf_avg_metrics, rf_std_metrics, rf_all_metrics = evaluate_with_time_series_cv(
        rf_model, X_scaled, y_scaled
    )

    print("\nEvaluating LightGBM Model...")
    lgb_avg_metrics, lgb_std_metrics, lgb_all_metrics = evaluate_with_time_series_cv(
        lgb_model, X_scaled, y_scaled
    )

    # 打印最终的评估结果
    print("\n=== Final Model Performance ===")
    print("\nRandom Forest Average Metrics:")
    for metric, value in rf_avg_metrics.items():
        print(f"{metric}: {value:.4f} ± {rf_std_metrics[metric]:.4f}")

    print("\nLightGBM Average Metrics:")
    for metric, value in lgb_avg_metrics.items():
        print(f"{metric}: {value:.4f} ± {lgb_std_metrics[metric]:.4f}")

    # 保存评估结果
    results_df = pd.DataFrame({
        'Model': ['Random Forest'] * len(rf_all_metrics) + ['LightGBM'] * len(lgb_all_metrics),
        'Fold': [m['fold'] for m in rf_all_metrics + lgb_all_metrics],
        'MAE': [m['MAE'] for m in rf_all_metrics + lgb_all_metrics],
        'MSE': [m['MSE'] for m in rf_all_metrics + lgb_all_metrics],
        'RMSE': [m['RMSE'] for m in rf_all_metrics + lgb_all_metrics],
        'R2': [m['R2'] for m in rf_all_metrics + lgb_all_metrics],
        'MAPE': [m['MAPE'] for m in rf_all_metrics + lgb_all_metrics]
    })
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print("\nEvaluation results have been saved to 'model_evaluation_results.csv'")
    print("\nGenerating performance visualizations...")

    # 获取预测值（使用全量训练数据的预测结果）
    rf_pred = scaler_y.inverse_transform(rf_model.predict(X_scaled).reshape(-1, 1)).ravel()
    lgb_pred = scaler_y.inverse_transform(lgb_model.predict(X_scaled).reshape(-1, 1)).ravel()
    ensemble_pred = scaler_y.inverse_transform(
        (rf_model.predict(X_scaled) * 0.6 + lgb_model.predict(X_scaled) * 0.4).reshape(-1, 1)
    ).ravel()

    # 调用三个可视化函数
    create_performance_visualizations(
        train_data_cleaned,  # 清洗后的训练数据
        y,  # 实际值
        rf_pred,  # Random Forest预测值
        lgb_pred,  # LightGBM预测值
        ensemble_pred,  # 集成模型预测值
        results_df  # 模型评估结果DataFrame
    )

    # 可视化特征重要性
    visualize_feature_importance(rf_model, lgb_model, feature_columns)

    # 可视化季节性模式
    plot_seasonal_patterns(train_data_cleaned)

    print("Visualizations have been saved to current directory.")
    # 继续原有的预测流程...
    print("\nProceeding with final model training and prediction...")

    # 使用全量数据训练最终模型
    rf_model.fit(X_scaled, y_scaled)
    lgb_model.fit(X_scaled, y_scaled)

    # 处理测试文件和生成预测（保持原有的预测代码不变）
    test_file_names = ['February', 'April', 'June', 'August', 'October', 'December']
    all_predictions = []

    for file_name in test_file_names:
        print(f"\nProcessing test file: {file_name}")
        test_data = pd.read_csv(f'./Test/{file_name}.csv')
        test_data = preprocess_data(test_data, is_train=False)

        last_24h_data = test_data.iloc[-24:]
        X_test = process_features(last_24h_data, feature_columns)
        X_test_scaled = scaler_X.transform(X_test)

        rf_pred_scaled = rf_model.predict(X_test_scaled)
        lgb_pred_scaled = lgb_model.predict(X_test_scaled)
        ensemble_pred_scaled = (rf_pred_scaled * 0.6 + lgb_pred_scaled * 0.4)
        ensemble_pred = scaler_y.inverse_transform(ensemble_pred_scaled.reshape(-1, 1)).ravel()

        all_predictions.extend(ensemble_pred)

    # 保存预测结果
    y_pred_test = np.array(all_predictions)
    Answer = pd.read_csv('./Test/Answer.csv')
    Answer['Load'] = y_pred_test
    Answer.to_csv('./Test/Answer.csv', index=False)


if __name__ == "__main__":
    main()
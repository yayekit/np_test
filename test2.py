import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def prepare_data(df):
    # Конвертуємо часові мітки
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    df['first_purchase_time'] = pd.to_datetime(df['first_purchase_time'])
    df['cohort_week'] = pd.to_datetime(df['cohort_week'])
    
    # Витягуємо ціну з product_id
    df['price'] = df['product_id'].str.extract(r'(\d+\.\d+)').astype(float)
    
    return df

def calculate_actual_ltv(df):
    # Групуємо дані по когортах та тижнях життя
    cohort_data = df.groupby(['cohort_week', 'lifetime_weeks'])['price'].sum().reset_index()
    
    # Підраховуємо кількість користувачів у кожній когорті
    cohort_sizes = df.groupby('cohort_week')['user_id'].nunique()
    
    # Розраховуємо LTV на користувача
    cohort_data['cohort_size'] = cohort_data['cohort_week'].map(cohort_sizes)
    cohort_data['ltv_per_user'] = cohort_data['price'] / cohort_data['cohort_size']
    
    # Розраховуємо кумулятивний LTV
    cohort_pivot = cohort_data.pivot(index='cohort_week', 
                                   columns='lifetime_weeks', 
                                   values='ltv_per_user').fillna(0)
    
    cumulative_ltv = cohort_pivot.cumsum(axis=1)
    
    return cumulative_ltv

def calculate_confidence_intervals(actual_ltv, confidence=0.95):
    # Розраховуємо стандартне відхилення для кожного тижня
    weekly_std = actual_ltv.std()
    weekly_mean = actual_ltv.mean()
    n = len(actual_ltv)
    
    # Розраховуємо margin of error
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin_error = z_score * (weekly_std / np.sqrt(n))
    
    lower_bound = weekly_mean - margin_error
    upper_bound = weekly_mean + margin_error
    
    return lower_bound, upper_bound

def predict_ltv(actual_ltv, weeks_to_predict=52):
    # Розраховуємо середні значення приросту LTV по тижнях
    avg_weekly_growth = actual_ltv.mean()
    
    # Створюємо DataFrame для предиктів
    predicted_ltv = pd.DataFrame(index=actual_ltv.index)
    
    # Заповнюємо відомі значення
    for week in actual_ltv.columns:
        predicted_ltv[week] = actual_ltv[week]
    
    # Розраховуємо середній приріст за останні 4 тижні для екстраполяції
    last_known_week = max(actual_ltv.columns)
    recent_growth = np.diff(avg_weekly_growth[-4:])
    last_weeks_growth = recent_growth.mean() if len(recent_growth) > 0 else 0
    
    # Предиктимо майбутні значення з затуханням
    decay_factor = 0.95  # Фактор затухання
    for week in range(last_known_week + 1, weeks_to_predict):
        if week - 1 in predicted_ltv.columns:
            growth = last_weeks_growth * (decay_factor ** (week - last_known_week))
            predicted_ltv[week] = predicted_ltv[week - 1] + max(growth, 0)  # Не допускаємо від'ємного росту
    
    return predicted_ltv

def plot_ltv_curves(actual_ltv, predicted_ltv):
    plt.figure(figsize=(15, 8))
    
    # Розраховуємо середні значення
    actual_mean = actual_ltv.mean()
    predicted_mean = predicted_ltv.mean()
    
    # Розраховуємо довірчі інтервали
    lower_bound, upper_bound = calculate_confidence_intervals(actual_ltv)
    
    # Plotting actual values
    plt.plot(actual_mean.index, actual_mean.values, 
             label='Actual LTV', marker='o', color='#2E86C1', linewidth=2)
    
    # Plotting confidence intervals for actual data
    plt.fill_between(actual_mean.index, 
                    lower_bound, 
                    upper_bound, 
                    alpha=0.2, 
                    color='#2E86C1', 
                    label='95% Confidence Interval')
    
    # Plotting predicted values
    last_actual_week = max(actual_mean.index)
    future_weeks = [w for w in predicted_mean.index if w > last_actual_week]
    
    if future_weeks:
        plt.plot(future_weeks, predicted_mean[future_weeks], 
                label='Predicted LTV', linestyle='--', color='#E67E22', linewidth=2)
        
        # Calculate prediction interval for future values
        std_actual = actual_ltv.std().mean()  # Use mean of standard deviations
        prediction_margin = 2 * std_actual  # Wider interval for predictions
        
        plt.fill_between(future_weeks,
                        predicted_mean[future_weeks] - prediction_margin,
                        predicted_mean[future_weeks] + prediction_margin,
                        alpha=0.1,
                        color='#E67E22',
                        label='Prediction Interval')
    
    plt.title('Average LTV per User Over Time', fontsize=14, pad=20)
    plt.xlabel('Weeks', fontsize=12)
    plt.ylabel('Cumulative LTV ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Додаємо анотації
    max_actual_ltv = actual_mean.max()
    plt.annotate(f'Max Actual LTV: ${max_actual_ltv:.2f}',
                xy=(last_actual_week, max_actual_ltv),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10)
    
    if future_weeks:
        max_predicted_ltv = predicted_mean[future_weeks].max()
        plt.annotate(f'Predicted Year 1 LTV: ${max_predicted_ltv:.2f}',
                    xy=(max(future_weeks), max_predicted_ltv),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(actual_ltv, predicted_ltv):
    # Розраховуємо метрики якості предикту для спільних тижнів
    common_weeks = sorted(set(actual_ltv.columns) & set(predicted_ltv.columns))
    actual_values = actual_ltv[common_weeks].mean()
    predicted_values = predicted_ltv[common_weeks].mean()
    
    # Розраховуємо метрики
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
    r2 = 1 - (np.sum((actual_values - predicted_values) ** 2) / 
              np.sum((actual_values - actual_values.mean()) ** 2))
    
    print("\nModel Evaluation Metrics:")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")
    print(f"R-squared (R²): {r2:.3f}")
    
    # Додатковий аналіз
    print("\nLTV Analysis:")
    print(f"Last Known Actual LTV: ${actual_values.iloc[-1]:.2f}")
    last_predicted = predicted_ltv[max(predicted_ltv.columns)].mean()
    print(f"Predicted Year 1 LTV: ${last_predicted:.2f}")
    
    weekly_growth = np.diff(actual_values)
    if len(weekly_growth) > 0:
        print(f"Average Weekly Growth: ${weekly_growth.mean():.2f}")
    
    return {
        'mape': mape,
        'rmse': rmse,
        'r2': r2
    }

def main():
    try:
        # Читаємо дані
        print("Loading and preparing data...")
        df = pd.read_csv('task2.csv')
        df = prepare_data(df)
        
        # Розрахунок фактичного LTV
        print("Calculating actual LTV...")
        actual_ltv = calculate_actual_ltv(df)
        
        # Предикт LTV
        print("Predicting future LTV...")
        predicted_ltv = predict_ltv(actual_ltv)
        
        # Візуалізація
        print("Generating visualization...")
        plot_ltv_curves(actual_ltv, predicted_ltv)
        
        # Оцінка якості моделі
        print("Evaluating model performance...")
        metrics = evaluate_model(actual_ltv, predicted_ltv)
        
        return actual_ltv, predicted_ltv, metrics
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    actual_ltv, predicted_ltv, metrics = main()
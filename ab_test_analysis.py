import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(file_path):
    """
    Завантаження та початкова підготовка даних
    """
    # Читаємо CSV файл
    df = pd.read_csv(file_path)
    
    # Конвертуємо дату в datetime
    df['install_date'] = pd.to_datetime(df['install_date'])
    
    return df

def calculate_metrics(df):
    """
    Розрахунок основних метрик по групах
    """
    metrics = {}
    
    for group in ['control', 'treatment']:
        group_data = df[df['test_group'] == group]
        total_users = len(group_data)
        
        metrics[group] = {
            'total_users': total_users,
            'trial_rate': (group_data['trial'] == 1).mean() * 100,
            'conversion_rate': (group_data['paid'] == 1).mean() * 100,
            'trial_to_paid': (group_data[group_data['trial'] == 1]['paid'] == 1).mean() * 100,
            'arpu': group_data['revenue_1m'].mean(),
            'arppu': group_data[group_data['paid'] == 1]['revenue_1m'].mean()
        }
    
    return pd.DataFrame(metrics)

def statistical_tests(df):
    """
    Проведення статистичних тестів
    """
    results = {}
    
    # Для конверсії в trial
    control_trial = df[df['test_group'] == 'control']['trial']
    treatment_trial = df[df['test_group'] == 'treatment']['trial']
    results['trial_test'] = stats.chi2_contingency(pd.crosstab(df['test_group'], df['trial']))
    
    # Для конверсії в paid
    control_paid = df[df['test_group'] == 'control']['paid']
    treatment_paid = df[df['test_group'] == 'treatment']['paid']
    results['paid_test'] = stats.chi2_contingency(pd.crosstab(df['test_group'], df['paid']))
    
    # Для revenue
    control_revenue = df[df['test_group'] == 'control']['revenue_1m']
    treatment_revenue = df[df['test_group'] == 'treatment']['revenue_1m']
    results['revenue_test'] = stats.mannwhitneyu(control_revenue, treatment_revenue)
    
    return results

def analyze_by_country(df):
    """
    Аналіз метрик по країнах
    """
    metrics_by_country = df.groupby(['country', 'test_group']).agg({
        'user_id': 'count',
        'trial': 'mean',
        'paid': 'mean',
        'revenue_1m': 'mean'
    }).round(4)
    
    metrics_by_country = metrics_by_country.rename(columns={
        'user_id': 'total_users',
        'trial': 'trial_rate',
        'paid': 'conversion_rate',
        'revenue_1m': 'arpu'
    })
    
    return metrics_by_country

def plot_key_metrics(df):
    """
    Візуалізація ключових метрик
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Trial Rate
    sns.barplot(data=df, x='test_group', y='trial', ax=axes[0,0])
    axes[0,0].set_title('Trial Rate by Group')
    axes[0,0].set_ylabel('Trial Rate')
    
    # Conversion Rate
    sns.barplot(data=df, x='test_group', y='paid', ax=axes[0,1])
    axes[0,1].set_title('Conversion Rate by Group')
    axes[0,1].set_ylabel('Conversion Rate')
    
    # ARPU
    sns.barplot(data=df, x='test_group', y='revenue_1m', ax=axes[1,0])
    axes[1,0].set_title('ARPU by Group')
    axes[1,0].set_ylabel('Revenue')
    
    # Trial to Paid Rate by Country
    trial_to_paid = df[df['trial'] == 1].groupby(['country', 'test_group'])['paid'].mean()
    trial_to_paid = trial_to_paid.unstack()
    trial_to_paid.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Trial to Paid Rate by Country')
    axes[1,1].set_ylabel('Trial to Paid Rate')
    
    plt.tight_layout()
    return fig

def run_analysis(file_path):
    """
    Запуск повного аналізу
    """
    # Завантаження даних
    df = load_and_prepare_data(file_path)
    
    # Розрахунок основних метрик
    metrics = calculate_metrics(df)
    
    # Статистичні тести
    stats_results = statistical_tests(df)
    
    # Аналіз по країнах
    country_metrics = analyze_by_country(df)
    
    # Візуалізація
    plots = plot_key_metrics(df)
    
    return {
        'metrics': metrics,
        'statistical_tests': stats_results,
        'country_metrics': country_metrics,
        'plots': plots
    }
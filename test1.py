import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def load_and_prepare_data(data):
    """
    Завантаження та базова підготовка даних
    """
    # Конвертуємо дати
    data['install_date'] = pd.to_datetime(data['install_date'])
    
    # Заповнюємо пропущені значення в revenue_1m нулями
    data['revenue_1m'] = data['revenue_1m'].fillna(0)
    
    return data

def calculate_metrics(data):
    """
    Розрахунок основних метрик по групах
    """
    metrics = {}
    for group in ['control', 'treatment']:
        group_data = data[data['test_group'] == group]
        total_users = len(group_data)
        
        # Конверсія в trial
        trial_users = len(group_data[group_data['trial'] == 1])
        trial_conv = trial_users / total_users
        
        # Конверсія з trial в paid
        trial_to_paid = len(group_data[(group_data['trial'] == 1) & (group_data['paid'] == 1)])
        paid_conv = trial_to_paid / trial_users if trial_users > 0 else 0
        
        # Загальна конверсія в paid
        total_conv = trial_to_paid / total_users
        
        # Середній revenue на користувача (ARPU)
        arpu = group_data['revenue_1m'].mean()
        
        metrics[group] = {
            'total_users': total_users,
            'trial_conv': trial_conv,
            'paid_conv': paid_conv,
            'total_conv': total_conv,
            'arpu': arpu
        }
    
    return metrics

def statistical_analysis(data):
    """
    Проведення статистичних тестів
    """
    results = {}
    
    # Chi-square test for trial conversion
    trial_contingency = pd.crosstab(data['test_group'], data['trial'])
    chi2, p_value_trial = chi2_contingency(trial_contingency)[:2]
    results['trial_conversion'] = {'chi2': chi2, 'p_value': p_value_trial}
    
    # Chi-square test for paid conversion
    trial_users = data[data['trial'] == 1]
    paid_contingency = pd.crosstab(trial_users['test_group'], trial_users['paid'])
    chi2, p_value_paid = chi2_contingency(paid_contingency)[:2]
    results['paid_conversion'] = {'chi2': chi2, 'p_value': p_value_paid}
    
    # T-test for revenue
    control_revenue = data[data['test_group'] == 'control']['revenue_1m']
    treatment_revenue = data[data['test_group'] == 'treatment']['revenue_1m']
    t_stat, p_value_revenue = stats.ttest_ind(control_revenue, treatment_revenue)
    results['revenue'] = {'t_stat': t_stat, 'p_value': p_value_revenue}
    
    return results

def analyze_by_country(data):
    """
    Аналіз метрик по країнах
    """
    country_metrics = {}
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        country_metrics[country] = calculate_metrics(country_data)
    
    return country_metrics

def plot_key_metrics(metrics):
    """
    Візуалізація ключових метрик
    """
    # Створюємо фігуру з підграфіками
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Конвертуємо метрики в формат для візуалізації
    metric_names = ['trial_conv', 'paid_conv', 'total_conv', 'arpu']
    control_values = [metrics['control'][m] for m in metric_names]
    treatment_values = [metrics['treatment'][m] for m in metric_names]
    
    # Графіки для кожної метрики
    titles = ['Trial Conversion', 'Paid Conversion', 'Total Conversion', 'ARPU']
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.bar(['Control', 'Treatment'], [control_values[i], treatment_values[i]])
        ax.set_title(title)
        ax.set_ylabel('Value')
        
    plt.tight_layout()
    plt.show()

def run_analysis(data):
    """
    Запуск повного аналізу
    """
    # Підготовка даних
    data = load_and_prepare_data(data)
    
    # Розрахунок основних метрик
    metrics = calculate_metrics(data)
    print("\nОсновні метрики:")
    for group, group_metrics in metrics.items():
        print(f"\n{group.upper()}:")
        for metric, value in group_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Статистичний аналіз
    stats_results = statistical_analysis(data)
    print("\nРезультати статистичних тестів:")
    for test, results in stats_results.items():
        print(f"\n{test}:")
        for stat, value in results.items():
            print(f"{stat}: {value:.4f}")
    
    # Аналіз по країнах
    country_metrics = analyze_by_country(data)
    print("\nМетрики по країнах:")
    for country, metrics in country_metrics.items():
        print(f"\n{country}:")
        for group, group_metrics in metrics.items():
            print(f"\n  {group.upper()}:")
            for metric, value in group_metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    # Візуалізація
    plot_key_metrics(metrics)


df = pd.read_csv('task1.csv')

run_analysis(df)
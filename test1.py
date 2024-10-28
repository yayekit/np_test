# Імпортуємо всі необхідні функції
from ab_test_analysis import *

# Запускаємо аналіз
results = run_analysis('task1.csv')

# Дивимось основні метрики
print("\nОсновні метрики:")
print(results['metrics'])

# Дивимось результати статистичних тестів
print("\nРезультати статистичних тестів:")
print("Trial Test p-value:", results['statistical_tests']['trial_test'][1])
print("Paid Test p-value:", results['statistical_tests']['paid_test'][1])
print("Revenue Test p-value:", results['statistical_tests']['revenue_test'].pvalue)

# Видача метрик по країнах
print("\nМетрики по країнах:")
print(results['country_metrics'])   
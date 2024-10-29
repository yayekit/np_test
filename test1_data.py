import pandas as pd
import numpy as np
from scipy import stats

# Read the data
df = pd.read_csv('task1.csv')

# 1. Overall group sizes
group_sizes = df.groupby('test_group').size()
print("\nGroup sizes:")
print(group_sizes)

# 2. Trial conversion rate analysis
trial_conv = df.groupby('test_group').agg({
    'trial': ['count', 'sum', lambda x: (x.sum()/x.count())*100]
}).round(2)
trial_conv.columns = ['total_users', 'trials', 'trial_conv_rate']
print("\nTrial conversion analysis:")
print(trial_conv)

# Statistical test for trial conversion
control_trial = df[df['test_group'] == 'control']['trial']
treatment_trial = df[df['test_group'] == 'treatment']['trial']
trial_stat_test = stats.chi2_contingency([
    [control_trial.sum(), len(control_trial) - control_trial.sum()],
    [treatment_trial.sum(), len(treatment_trial) - treatment_trial.sum()]
])
print("\nTrial conversion Chi-square p-value:", trial_stat_test[1])

# 3. Paid conversion analysis for trials
trial_users = df[df['trial'] == 1]
paid_conv = trial_users.groupby('test_group').agg({
    'paid': ['count', 'sum', lambda x: (x.sum()/x.count())*100]
}).round(2)
paid_conv.columns = ['trial_users', 'paid_users', 'paid_conv_rate']
print("\nPaid conversion analysis (among trials):")
print(paid_conv)

# p-value for conversions
control_paid = trial_users[trial_users['test_group'] == 'control']['paid']
treatment_paid = trial_users[trial_users['test_group'] == 'treatment']['paid']
if len(control_paid) > 0 and len(treatment_paid) > 0:
    paid_stat_test = stats.chi2_contingency([
        [control_paid.sum(), len(control_paid) - control_paid.sum()],
        [treatment_paid.sum(), len(treatment_paid) - treatment_paid.sum()]
    ])
    print("\nPaid conversion Chi-square p-value:", paid_stat_test[1])

# 4. Revenue analysis
revenue_analysis = df.groupby('test_group').agg({
    'revenue_1m': ['count', 'sum', 'mean']
}).round(2)
revenue_analysis.columns = ['users', 'total_revenue', 'arpu']
print("\nRevenue analysis:")
print(revenue_analysis)

# p-value for revenue
control_revenue = df[df['test_group'] == 'control']['revenue_1m']
treatment_revenue = df[df['test_group'] == 'treatment']['revenue_1m']
revenue_stat_test = stats.mannwhitneyu(control_revenue, treatment_revenue)
print("\nRevenue Mann-Whitney U test p-value:", revenue_stat_test.pvalue)

# 5. Country breakdown
country_analysis = df.groupby(['country', 'test_group']).agg({
    'trial': ['count', 'sum', lambda x: (x.sum()/x.count())*100],
    'paid': 'sum',
    'revenue_1m': ['sum', 'mean']
}).round(2)
print("\nCountry breakdown:")
print(country_analysis)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-pastel')
colors = ['#2E86C1', '#E67E22']  # Blue and Orange

def create_slide3_conversion_metrics(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Trial Conversion
    trial_rates = df.groupby('test_group')['trial'].mean() * 100
    trial_bars = ax1.bar(range(2), trial_rates, color=colors)
    ax1.set_title('Trial Start Rate', pad=20)
    ax1.set_ylabel('Conversion Rate (%)')
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(['Monthly\n$5.99', 'Weekly\n$2.99'])
    
    for bar in trial_bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2.,
                f'{height:.1f}%',
                ha='center', va='center', color='white', fontsize=12)
    
    # Paid Conversion
    trial_users = df[df['trial'] == 1]
    paid_rates = trial_users.groupby('test_group')['paid'].mean() * 100
    paid_bars = ax2.bar(range(2), paid_rates, color=colors)
    ax2.set_title('Paid Conversion Rate (Trial Users)', pad=20)
    ax2.set_ylabel('Conversion Rate (%)')
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(['Monthly\n$5.99', 'Weekly\n$2.99'])
    
    for bar in paid_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height/2.,
                f'{height:.1f}%',
                ha='center', va='center', color='white', fontsize=12)
    
    plt.suptitle('Conversion Metrics Comparison', fontsize=14, y=1.05)
    plt.tight_layout()
    return fig

def create_slide4_revenue_impact(df):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    control_total_users = len(df[df['test_group'] == 'control'])
    treatment_total_users = len(df[df['test_group'] == 'treatment'])
    control_paying_users = len(df[(df['test_group'] == 'control') & (df['paid'] == 1)])
    treatment_paying_users = len(df[(df['test_group'] == 'treatment') & (df['paid'] == 1)])
    control_revenue = df[df['test_group'] == 'control']['revenue_1m'].sum()
    treatment_revenue = df[df['test_group'] == 'treatment']['revenue_1m'].sum()
    
    metrics = {
        'Monthly $5.99': {
            'Overall ARPU': control_revenue / control_total_users,
            'Paying ARPU': control_revenue / control_paying_users if control_paying_users > 0 else 0,
            'Total Revenue': control_revenue,
            'Paying Users': control_paying_users
        },
        'Weekly $2.99': {
            'Overall ARPU': treatment_revenue / treatment_total_users,
            'Paying ARPU': treatment_revenue / treatment_paying_users if treatment_paying_users > 0 else 0,
            'Total Revenue': treatment_revenue,
            'Paying Users': treatment_paying_users
        }
    }
    
    # Plot 1: ARPUs
    x = np.arange(2)
    width = 0.35
    
    overall_arpu = [metrics['Monthly $5.99']['Overall ARPU'], 
                   metrics['Weekly $2.99']['Overall ARPU']]
    paying_arpu = [metrics['Monthly $5.99']['Paying ARPU'], 
                  metrics['Weekly $2.99']['Paying ARPU']]
    
    bars1 = ax1.bar(x - width/2, overall_arpu, width, 
                   label='Overall ARPU', color=colors[0])
    bars2 = ax1.bar(x + width/2, paying_arpu, width, 
                   label='Paying Users ARPU', color=colors[1])
    
    ax1.set_title('Average Revenue Per User')
    ax1.set_ylabel('ARPU ($)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Monthly\n$5.99', 'Weekly\n$2.99'])
    ax1.legend()
    
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}',
                    ha='center', va='bottom')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Plot 2: Total Revenue
    total_rev = [metrics['Monthly $5.99']['Total Revenue'], 
                 metrics['Weekly $2.99']['Total Revenue']]
    bars3 = ax2.bar([0, 1], total_rev, color=colors[0])
    ax2.set_title('Total Revenue')
    ax2.set_ylabel('Revenue ($)')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Monthly\n$5.99', 'Weekly\n$2.99'])
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height/2.,
                f'${height:.2f}',
                ha='center', va='center', color='white', fontsize=12)
    
    # Plot 3: Paying Users
    users = [metrics['Monthly $5.99']['Paying Users'], 
             metrics['Weekly $2.99']['Paying Users']]
    bars4 = ax3.bar([0, 1], users, color=colors[0])
    ax3.set_title('Number of Paying Users')
    ax3.set_ylabel('Count')
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Monthly\n$5.99', 'Weekly\n$2.99'])
    
    for bar in bars4:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height/2.,
                f'{int(height)}',
                ha='center', va='center', color='white', fontsize=12)
    
    plt.suptitle('Revenue Metrics Comparison', fontsize=14, y=1.05)
    plt.tight_layout()
    return fig

    # Plot 3: By country
def create_slide5_geographic(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    countries = ['US', 'GB', 'AU', 'CA']
    country_names = {'US': 'United States', 'GB': 'United Kingdom', 
                    'AU': 'Australia', 'CA': 'Canada'}
    
    x = np.arange(len(countries))
    width = 0.35
    
    control_arpu = []
    treatment_arpu = []
    
    for country in countries:
        control_arpu.append(df[(df['country'] == country) & 
                             (df['test_group'] == 'control')]['revenue_1m'].mean())
        treatment_arpu.append(df[(df['country'] == country) & 
                               (df['test_group'] == 'treatment')]['revenue_1m'].mean())
    
    ax.bar(x - width/2, control_arpu, width, label='Monthly $5.99', color=colors[0])
    ax.bar(x + width/2, treatment_arpu, width, label='Weekly $2.99', color=colors[1])
    
    ax.set_ylabel('Average Revenue per User ($)')
    ax.set_title('Revenue Performance by Country')
    ax.set_xticks(x)
    ax.set_xticklabels([country_names[c] for c in countries])
    ax.legend()
    
    rects1 = ax.patches[:len(countries)]
    rects2 = ax.patches[len(countries):]
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                f'${height:.2f}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig

def main():
    df = pd.read_csv('task1.csv')
    
    slide3_fig = create_slide3_conversion_metrics(df)
    slide3_fig.savefig('slide3_conversion_metrics.png', bbox_inches='tight', dpi=300)
    
    slide4_fig = create_slide4_revenue_impact(df)
    slide4_fig.savefig('slide4_revenue_impact.png', bbox_inches='tight', dpi=300)
    
    slide5_fig = create_slide5_geographic(df)
    slide5_fig.savefig('slide5_geographic_analysis.png', bbox_inches='tight', dpi=300)
    
    plt.close('all')

if __name__ == "__main__":
    main()
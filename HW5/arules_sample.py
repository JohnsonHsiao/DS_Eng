import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules

org_df = pd.read_csv("amr_horse_ds.csv")

age_bins = [0, 1, 5, 10, 20, 30] # age binning
age_labels = ['Infant', 'Young', 'Adolescent', 'Adult', 'Senior']  # labels for age bins
org_df['Age_binned'] = pd.cut(org_df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

# 2. One-hot encoding
df_encoded = pd.get_dummies(org_df.drop('Age', axis=1), columns=['Sex', 'Decade', 'Gram_ID', 'Age_binned'], drop_first=True)

# 3. different parameter combinations
min_sup_values = [0.05, 0.1, 0.4]
min_conf_values = [0.7, 0.85, 0.95]
min_lift_values = [1.1, 1.5, 4]

rule_count_summary = []

# 4. iterate
for min_sup in min_sup_values:
    for min_conf in min_conf_values:
        frequent_patterns = fpgrowth(df_encoded, min_support=min_sup, use_colnames=True)
        
        rules = association_rules(frequent_patterns, metric="confidence", min_threshold=min_conf)
        
        for min_lift in min_lift_values:

            filtered_rules = rules[rules['lift'] >= min_lift]
            
            rule_count = len(filtered_rules)
            
            rule_count_summary.append({
                'min_support': min_sup,
                'min_confidence': min_conf,
                'min_lift': min_lift,
                'rule_count': rule_count
            })

# 5. 
rule_count_df = pd.DataFrame(rule_count_summary)
filtered_df = rule_count_df[(rule_count_df['rule_count'] >= 20) & (rule_count_df['rule_count'] <= 50)]
print(filtered_df)

if not filtered_df.empty:
    best_params = filtered_df.iloc[0]  
    frequent_patterns_best = fpgrowth(df_encoded, min_support=best_params['min_support'], use_colnames=True)
    rules_best = association_rules(frequent_patterns_best, metric="confidence", min_threshold=best_params['min_confidence'])
    high_lift_rules_best = rules_best[rules_best['lift'] >= best_params['min_lift']]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(high_lift_rules_best['support'], high_lift_rules_best['confidence'], high_lift_rules_best['lift'], marker="*")
    ax.set_xlabel('Support')
    ax.set_ylabel('Confidence')
    ax.set_zlabel('Lift')
    plt.show()

    high_lift_rules_best.to_csv('filtered_arules.csv')
    print(high_lift_rules_best)

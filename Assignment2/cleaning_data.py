import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
start_time = time.time()

df = pd.read_csv('training_set_VU_DM.csv', sep=',')

df.replace('NULL', np.nan, inplace=True)


#impute
df['prop_review_score'].fillna(df['prop_starrating'], inplace=True)

df['prop_location_score2'].fillna(df.groupby('prop_id')['prop_location_score2'].transform('first'), inplace=True)
worst_score = df.groupby('srch_id')['prop_location_score2'].transform('min')
df['prop_location_score2'].fillna(worst_score, inplace=True)
df['prop_location_score2'].fillna(0.001, inplace=True)



'''
columns_for_training = ['prop_id', 'visitor_location_country_id', 'site_id',
                        'prop_starrating','prop_review_score','prop_location_score1','prop_location_score2',
                        'srch_destination_id','srch_length_of_stay','srch_room_count','srch_children_count','srch_adults_count'] #for now knn for orig_destination_distance
column_to_impute = 'orig_destination_distance'
selected_data = df[columns_for_training + [column_to_impute]].copy()
missing_values_data = selected_data[selected_data[column_to_impute].isnull()].copy()
complete_values_data = selected_data[selected_data[column_to_impute].notnull()].copy()
imputer = KNNImputer()
imputer.fit(complete_values_data)
imputed_values = pd.DataFrame(imputer.fit_transform(missing_values_data), columns=missing_values_data.columns)
final_data = pd.concat([imputed_values, complete_values_data], axis=0)
df[column_to_impute] = final_data[column_to_impute]
'''





missing_percentages = df.isnull().mean() * 100
print(missing_percentages)

non_zero_percentages = missing_percentages[missing_percentages != 0]
sorted_percentages = non_zero_percentages.sort_values()
colors = plt.cm.Reds(sorted_percentages)

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
sorted_percentages.plot(kind='bar', color=colors)
plt.ylabel('Missing Percentage')
plt.xticks(rotation = 90, fontsize=8)  # Rotate x-axis labels if needed
plt.grid(axis='y', linestyle='--', color='gray', alpha = 0.5)
plt.show()



#removing columns
#columns_to_remove = missing_percentages[missing_percentages > 50].index
#columns_to_remove = columns_to_remove.drop('gross_bookings_usd') # not removing this one
#df = df.drop(columns=columns_to_remove)

missing_rows = df[df['prop_review_score'].isnull()]
print(missing_rows)



df.to_csv('edited_training.csv', index=False)
df_subset = df.head(50000)
df_subset.to_csv('sample_edited_training.csv', index=False)




end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))


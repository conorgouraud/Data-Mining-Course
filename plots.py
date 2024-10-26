import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

df = pd.read_csv('edited_training.csv', sep=',')
#df = pd.read_csv('sample_edited_training.csv', sep=',')


#ranking by price
df['price_order'] = df.groupby('srch_id')['price_usd'].rank()

#######

df['fraction_booked'] = df.groupby('prop_id')['click_bool'].transform(lambda x: x.sum() / x.count() if x.count() >= 5 else 0)
#print(df['fraction_booked'])

#######

print(df.iloc[1168578])

value_counts = df['srch_id'].value_counts()
clicked_props = df[df['click_bool'] == 1].copy()
top_ten_rows = clicked_props.nlargest(10, 'price_usd')['price_usd']
print(top_ten_rows)

booked_props = df[df['booking_bool'] == 1].copy()
top_ten_rows = booked_props.nlargest(10, 'price_usd')['price_usd']
print(top_ten_rows)

mean_count = value_counts.mean()
print('average amount of hotels per search =', mean_count)
print(df['price_order'].max())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
b = 37
ax1.hist(df['price_order'], bins=b, density=True, alpha=0.4, label='All Properties')
ax1.hist(clicked_props['price_order'], bins=b, density=True, alpha=0.4, label='Clicked Properties')
ax1.set_xlabel('Price Order', fontsize=14)
ax1.set_ylabel('Density', fontsize=16)
ax1.legend()

ax2.hist(df['price_order'], bins=b, density=True, alpha=0.4, label='All Properties')
ax2.hist(booked_props['price_order'], bins=b, density=True, alpha=0.4, label='Booked Properties')
ax2.set_xlabel('Price Order', fontsize=14)
ax2.set_yticklabels([])
ax2.legend()

plt.tight_layout()
plt.show()

######
'''
unique_searches = df.drop_duplicates(subset=['srch_id'])
unique_hotels = df.drop_duplicates(subset=['prop_id'])
ordered_hotels = unique_searches.sort_values('srch_length_of_stay', ascending=False)

plt.scatter(unique_hotels['prop_location_score1'], unique_hotels['prop_location_score2'], s = 0.1)
plt.show()

unique_hotels = unique_hotels.dropna(subset=['prop_location_score1', 'prop_location_score2'])
plt.hist2d(unique_hotels['prop_location_score1'], unique_hotels['prop_location_score2'], bins=28, cmap= 'Blues')
plt.show()

unique_hotels = unique_hotels.dropna(subset=['prop_review_score', 'prop_starrating'])
plt.hist2d(unique_hotels['prop_review_score'], unique_hotels['prop_starrating'], bins=28, cmap= 'Blues')
plt.colorbar()
plt.show()

######

booking_fraction = df.groupby('prop_review_score')['booking_bool'].mean()
clicked_fraction = df.groupby('prop_review_score')['click_bool'].mean()

x_pos = np.arange(len(booking_fraction))
bar_width = 0.6
color_map = plt.cm.get_cmap('tab10')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
booking_bars = ax1.bar(x_pos, booking_fraction, width=bar_width, color=color_map(range(len(booking_fraction))))
ax1.set_xlabel('Review Score', fontsize=14)
ax1.set_ylabel('Fraction', fontsize=14)
ax1.set_title('Booked', fontsize=16)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(booking_fraction.index)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.grid(axis='y', linestyle='--')
clicked_bars = ax2.bar(x_pos, clicked_fraction, width=bar_width, color=color_map(range(len(clicked_fraction))))
ax2.set_xlabel('Review Score', fontsize=14)
ax2.set_title('Clicked', fontsize=16)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(clicked_fraction.index)
ax2.tick_params(axis='x', labelsize=12)
ax2.grid(axis='y', linestyle='--')
ax2.set_ylim(0, 0.055)
ax2.tick_params(axis='y', labelleft=False)
plt.tight_layout()
plt.show()
'''
########
'''
grouped = df.groupby(['random_bool', 'position'])['click_bool', 'booking_bool'].mean().reset_index()
df_0 = grouped[grouped['random_bool'] == 0]
df_1 = grouped[grouped['random_bool'] == 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
ax1.plot(df_0['position'], df_0['click_bool'], label='Random = 0 (click_bool)')
ax1.plot(df_1['position'], df_1['click_bool'], label='Random = 1 (click_bool)')
ax1.set_xlabel('Position', fontsize=14)
ax1.set_title('Clicked')
ax1.set_ylabel('Fraction', fontsize=14)
ax2.plot(df_0['position'], df_0['booking_bool'], label='Random = 0')
ax2.plot(df_1['position'], df_1['booking_bool'], label='Random = 1')
ax2.set_xlabel('Position', fontsize=14)
ax2.set_title('Booked')
ax2.legend()
ax1.grid(axis='y', linestyle='--', color='gray')
ax2.grid(axis='y', linestyle='--', color='gray')

plt.tight_layout()
plt.show()
'''

##############
'''
columns_to_plot = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
                   'price_usd', 'srch_length_of_stay','srch_booking_window']

subset_df = df[columns_to_plot]
data = [subset_df[column].dropna().values for column in columns_to_plot]

plt.figure(figsize=(8, 6))
box_plot = plt.boxplot(data, vert=False, patch_artist=True, flierprops={'marker': 'o', 'markersize': 1, 'markerfacecolor': 'black'})
#plt.boxplot(data, vert=False)
plt.xscale('log')
ytick_labels = [column for column in columns_to_plot]
ytick_positions = range(1, len(columns_to_plot) + 1)
plt.yticks(ytick_positions, ytick_labels)
for patch in box_plot['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.5)

plt.show()
'''
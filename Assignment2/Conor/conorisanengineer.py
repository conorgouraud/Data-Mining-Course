import pandas as pd
import numpy as np
import time

start_time = time.time()

#df = pd.read_csv('edited_training.csv', sep=',')
#df = pd.read_csv('sample_edited_training.csv', sep=',')

df = pd.read_csv('training_set_VU_DM.csv', sep=',')

df.replace('NULL', np.nan, inplace=True)

#imputing stuff (orig_destination_distance might also need to be imputed?)
lower_quartile = df.groupby('srch_id')['prop_starrating'].transform(lambda x: x.quantile(0.25))
df['prop_starrating'].fillna(lower_quartile, inplace=True)

df['prop_location_score2'].fillna(df.groupby('prop_id')['prop_location_score2'].transform('first'), inplace=True)
lower_quartile = df.groupby('srch_id')['prop_location_score2'].transform(lambda x: x.quantile(0.25))
df['prop_location_score2'].fillna(lower_quartile, inplace=True)
df['prop_location_score2'].fillna(0.001, inplace=True)



#ranking columns
df['price_order'] = df.groupby('srch_id')['price_usd'].rank()
df['starrating_order'] = df.groupby('srch_id')['prop_starrating'].rank()
df['prop_location_score2_order'] = df.groupby('srch_id')['prop_location_score2'].rank()
df['prop_location_score1_order'] = df.groupby('srch_id')['prop_location_score1'].rank()



#combining some columns
df['hotel_quality'] = df['price_usd'] / df['prop_starrating']

df['hotel_quality_user'] = df['price_usd'] / df['prop_review_score']

df['star_diff'] = abs(df['prop_starrating'] - df['prop_review_score'])

df['usd_diff'] = abs(df['price_usd'] - df['visitor_hist_adr_usd'])


#normalisation by search id
max_value = df.groupby('srch_id')['prop_starrating'].transform('max')
df['prop_starrating_norm_bysearch'] = df['prop_starrating'] / max_value

max_value = df.groupby('srch_id')['prop_review_score'].transform('max')
df['prop_review_score_norm_bysearch'] = df['prop_review_score'] / max_value

max_value = df.groupby('srch_id')['prop_location_score1'].transform('max')
df['prop_location_score1_norm_bysearch'] = df['prop_location_score1'] / max_value

max_value = df.groupby('srch_id')['prop_location_score2'].transform('max')
df['prop_location_score2_norm_bysearch'] = df['prop_location_score2'] / max_value

max_value = df.groupby('srch_id')['prop_log_historical_price'].transform('max')
df['prop_log_historical_price_norm_bysearch'] = df['prop_log_historical_price'] / max_value

max_value = df.groupby('srch_id')['price_usd'].transform('max')
df['price_usd_norm_bysearch'] = df['price_usd'] / max_value

max_value = df.groupby('srch_id')['visitor_hist_adr_usd'].transform('max')
df['visitor_hist_adr_usd_norm_bysearch'] = df['visitor_hist_adr_usd'] / max_value



#normalised joint columns (same as previous 4, but normed)

max_value = df.groupby('srch_id')['hotel_quality'].transform('max')
df['hotel_quality_norm'] = df['hotel_quality'] / max_value

max_value = df.groupby('srch_id')['hotel_quality_user'].transform('max')
df['hotel_quality_user_norm'] = df['hotel_quality_user'] / max_value

max_value = df.groupby('srch_id')['star_diff'].transform('max')
df['star_diff_norm'] = df['star_diff'] / max_value

max_value = df.groupby('srch_id')['usd_diff'].transform('max')
df['usd_diff_norm'] = df['usd_diff'] / max_value



#normalisation by prop id
max_value = df.groupby('prop_id')['prop_starrating'].transform('max')
df['prop_starrating_norm_byprop'] = df['prop_starrating'] / max_value

max_value = df.groupby('prop_id')['prop_review_score'].transform('max')
df['prop_review_score_norm_byprop'] = df['prop_review_score'] / max_value

max_value = df.groupby('prop_id')['prop_location_score1'].transform('max')
df['prop_location_score1_norm_byprop'] = df['prop_location_score1'] / max_value

max_value = df.groupby('prop_id')['prop_location_score2'].transform('max')
df['prop_location_score2_norm_byprop'] = df['prop_location_score2'] / max_value

max_value = df.groupby('prop_id')['prop_log_historical_price'].transform('max')
df['prop_log_historical_price_norm_byprop'] = df['prop_log_historical_price'] / max_value

max_value = df.groupby('prop_id')['price_usd'].transform('max')
df['price_usd_norm_byprop'] = df['price_usd'] / max_value



#competitive columns
comp_inv_columns = ['comp1_inv','comp2_inv','comp3_inv','comp4_inv','comp5_inv','comp6_inv','comp7_inv','comp8_inv']
df['comp_inv_most_frequent'] = df[comp_inv_columns].mean(axis=1)

comp_rate_columns = ['comp1_rate','comp2_rate','comp3_rate','comp4_rate','comp5_rate','comp6_rate','comp7_rate','comp8_rate']
df['comp_rate_most_frequent'] = df[comp_rate_columns].mean(axis=1)
df['comp_rate_most_frequent'] = df['comp_rate_most_frequent'].apply(lambda x: -1 if x < 0 else 1 if x > 0 else 0)

df['comp_inv_most_frequent'].fillna(-100, inplace=True)
df['comp_rate_most_frequent'].fillna(-100, inplace=True)
df['comp_inv_most_frequent'] = df['comp_inv_most_frequent'].astype(int)


# time column stuff (month in the future)

df['date_time'] = pd.to_datetime(df['date_time'])
df['future_month'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'], unit='D')
df['future_month'] = df['future_month'].dt.month/12 #normalised too

df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour


#another feature from 2nd place

booked_props = df[df['booking_bool'] == 1].copy()
meanstar = np.mean(df['prop_starrating'])
df['prop_starrating_monotonic'] = abs(df['prop_starrating'] - meanstar)


#dropping columns that dont seem useful
df.drop(['srch_id',
'srch_saturday_night_bool',
'comp1_inv',
'comp3_inv',
'comp4_rate_percent_diff',
'comp6_rate',
'comp6_inv',
'comp7_rate',
'prop_starrating_norm_byprop',
'prop_review_score_norm_byprop','prop_location_score1_norm_byprop',
'comp_inv_most_frequent',
'comp4_inv',
'comp6_rate_percent_diff',
'comp7_inv',
'srch_destination_id',
'comp1_rate',
'comp1_rate_percent_diff',
'comp2_rate',
'comp2_inv',
'comp3_rate',
'comp4_rate',
'comp8_inv',
'srch_room_count',
'comp_rate_most_frequent',
'srch_adults_count',
'comp7_rate_percent_diff',
'comp5_inv'], axis=1, inplace=True)



df.to_csv('engineered_training.csv', index=False)


end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))
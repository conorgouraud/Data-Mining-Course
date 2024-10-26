import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Latex
import re
import math

# Import data
df = pd.read_csv('ODI-2023_new.csv', sep=';')

# Seperate column with "Bed times"
bed_time = df['Time you went to bed Yesterday']

# First, dots are replaced with colons at the beginning of time values and spaces are replaced with empty string

bed_time = bed_time.str.replace(r'\.|\s', lambda x: ':' if x.group(0) == '.' else '', regex=True)

# Regular expression (regex) to match time in hh:mm or h:mm format
time_reg = r'\b\d{1,2}:\d{2}\s*(?:[AaPp][Mm])?\b|\baround\s*(\d{1,2}:\d{2}\s*[AaPp][Mm])\b'

# Define a callback function to replace the matched time values
def replacer(match):
    time_str = match.group(0)
    if len(time_str) > 7:
        # Replace dots with colons at the beginning of time values
        time_str = time_str.replace('.', ':')
        # Replace spaces with empty string
        time_str = time_str.replace(' ', '')
    return time_str

# To find and replace the desired values in the column, we use the regex and the callback function
bed_time = bed_time.str.replace(time_reg, replacer, regex=True)


# Separate the hour and minute parts using regex
time_parts = bed_time.str.extract(r'(?P<hour>\d{1,2}):?(?P<minute>\d{2})?')

# Replace NaN values in minute column with '00'
time_parts['minute'] = time_parts['minute'].fillna('00')

# Fill the hour part with a 0 if there is a space in front
time_parts['hour'] = time_parts['hour'].str.zfill(2)

# Combine hour and minute parts to form HH:MM format
bed_time = time_parts['hour'] + ':' + time_parts['minute']

# Replace other non-sense values with NaN
invalid = ['^.*[^0-9:\sAaPpMm].*$', '^.{0,1}$']
bed_time = bed_time.replace(invalid, np.nan, regex=True)

# Convert remaining missing values to NaN
bed_time = bed_time.fillna(np.nan)


hour_part = time_parts['hour']


bed_time_list = bed_time.tolist()
# Check for NaN values
nan_values = pd.isna(bed_time_list)
num_missing = nan_values.sum()

# Convert nan to 'NaN'
bed_time_list = ['NaN' if isinstance(x, float) and np.isnan(x) else x for x in bed_time_list]

# We make a boolean mask that picks only these numbers
bed_time_mask = bed_time.str[:2].isin(['07', '08', '09', '10', '11']) & bed_time.notna()

print(bed_time_mask)
bed_time.loc[bed_time_mask] = bed_time.loc[bed_time_mask].apply(lambda x: str(int(x[:2]) + 12) + x[2:])

#Extract hour component from bedtime list, ignoring 'NaN' values
hour_list = []
for time in bed_time_list:
    if time != 'NaN':
        try:
            hour = int(time.split(':')[0])
            hour_list.append(hour)
        except ValueError:
            pass

# Create two histograms with overlapping bins
plt.hist(hour_list, bins = 10, edgecolor='black')
plt.xlabel('O clock')
plt.ylabel('Frequency')
plt.title('Bed Time distribution')
plt.show()

print(bed_time)

# Seperate column with "Bed times"
df['Time you went to bed Yesterday'] = bed_time

df.to_excel('cleaned_bed_times.xlsx', index=False)
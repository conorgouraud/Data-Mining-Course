import pandas as pd
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import re
from datetime import datetime
from sklearn.impute import KNNImputer

def magdas_bedtime_cleaning(dataframe):
    df = dataframe.copy(deep=True)

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

    #print(bed_time_mask)
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
    #plt.hist(hour_list, bins = 10, edgecolor='black')
    plt.xlabel('O clock')
    plt.ylabel('Frequency')
    plt.title('Bed Time distribution')
    #plt.show()

    #print(bed_time)

    # Seperate column with "Bed times"
    df['Time you went to bed Yesterday'] = bed_time

    return df


#df.to_excel('cleaned_bed_times.xlsx', index=False) #saved as excel and then use this for conor part

#######################################################################################################################
#df = pd.read_excel('cleaned_bed_times.xlsx')

def conorclean(dataframe):
    df = dataframe.copy(deep=True)
    nan = 'nan'
    #sorry bout this part
    new_column = pd.DataFrame({'B': ['23/11/1997', '08/05/1995', nan, '25-06-1996', '18-08-1998', '13-03-2000', '21-07-1999', '08-03-2000', '24-11-1996', '07.12.1998', '02-01-1999',
                                     '14/11/1999', 'joke', nan, nan, '25.09.2000', '04 02 1999', '27-11-1999', '29-12-1997', '09-02-2000', '09-05-1999', '14-05-1999', '11/11/1997',
                                     nan, '08-08-1997', '11-05-1998', '15-09-2002', '26/04/1997', '24-10-2000', '05.07.2002', '31-01-2000', '03/05/1997', '31-03-1999', nan, '18-12-1998',
                                     '17/02/1998', '22/08/1999', '28-09-1998', '5 07 2001', '12-03-1999', '14-12-1997', '31/10/2001', '10/01/2001', '25/01/1998', '24/04/1996',
                                     '08-08-1999', '18-06-1999', '19/09/1998', '08-09-2000', '17-04-1998', '02-11-2000', '06-09-1998', '28-2-2000', '30 03 1998', '12-06-1998',
                                     '16-01-1999', '22.04.1996', '12-08-1999', nan, '2/6/1997', '28-05-1999', '28 12 2000', '31-03-1998', '02-05-1999', '06/11/2000', '02-06-2000',
                                     nan, '19-02-2000', '07/08/1997', '01/05/1997', '31/03/1991', '08-07-2001', '17/6/1997', nan, '18-08-1996', nan, '04-12-1998', '16-09-1999', nan,
                                     '5/11/2000', '26/10/1996', nan, '23-10-2001', '22 06 1999', nan, '12-08-2000', '21/02/2000', '15-09-1999', '12.12.1997', '04.09.2000', 'joke',
                                     '23-08-1999', '31-08-1997', '23-05-2000', '24/11/1999', nan, nan, '27.09.1999', '02-05-2000', nan, '03/04/1998', 'joke', '10-09-1997', nan,
                                     '08-06-2000', '12.02.1997', nan, '30/04/1998', '15/07/1997', '17.04.1997', '14-06-1999', '30 05 1997', '6-8-1997', '06/05/1999', '04/03/2000',
                                     'joke', 'joke', '11-08-1997', '6/11/1999', nan, '19-05-2000', '14-08-2000', '5-2-1997', '14/07/1994', '23/4/1985', '30 09 1997', '22-12-1997',
                                     '13/03/1999', '11-05-1998', '25/04/1999', '25-05-1999', '05/12/1999', '20.01.1999', nan, '15-01-2001', nan, '11/08/1992', '03 07 1998', '25-04-1987',
                                     '08-08-2001', '17/01/1997', nan, nan, '15-11-1998', 'joke', '10/03/2001', nan, '15-01-1999', '07-10-1999', '07-07-1995', nan, '11-09-2000', '09-07-2000',
                                     '04/11/2000', '10-01-2001', '21-12-1999', '23.12.1999', '06-11-1999', '01/01/1990', '02.08.1998', '03-05-2000', '16/06/1999', '14-12-2000', '21/03/2000',
                                     '23-03-1998', nan, '23/11/1997', '18-11-2000', '08/05/1998', '05 01 1998', '09/03/1998', 'joke', '29/11/2000', nan, '31/03/1998', '10-05-2001', nan,
                                     '26/06/1999', '01/05/1995', '20/06/1999', nan, '10-10-2001', '18-08-1999', '29-06-2000', '11-09-1999', '18-11-2000', '21-09-1999', nan, '20/11/1990',
                                     'joke', '23.04.1995', '12-08-1999', nan, nan, '12/03/1999', '24-08-1999', '14-09-2001', '19-05-2000', nan, '20/09/1998', '28/10/1996', '25-11-2000',
                                     '05-05-1998', '05/11/1999', '01/06/1999', '23-08-1999', '05/02/1998', 'joke', '13-06-2000', '17-04-1998', '06/11/1997', '15-4-1999', '21-02-2000',
                                     nan, '09-09-2001', '09/10/2000', 'joke', '20.06.2000', '07-07-2001', nan, '04/02/1999', '11/03/1995', nan, '23-02-1999', nan, '19 09 1997',
                                     '30/12/2000', '04-01-2001', '15/09/1999', '11-12-1997', '14/05/1998', '05-09-2000', '27-11-2000', '30/12/1999', '10/12/1998', '29-05-2000',
                                     'joke', '03-01-2001', '19-11-1999', '20-02-2002', '16/01/2001', nan, '4/6/2000', 'joke', '06-02-1996', nan, '04-08-1998', nan, 'joke',
                                     '29 03 1998', 'joke', nan, '18.08.1997', 'joke', 'joke', nan, '31-12-1999', '27-10-1997', '13/07/1999', '08/08/1998', '09/09/1993',
                                     '29-06-2000', '26-10-2000', '22 11 1999', 'joke', '24-01-2000', '14-09-2000', nan, 'joke', '12-02-2001', '28 09 2000', nan, '15-01-1999',
                                     nan, '6-11-1997', '10-07-2000', '02/11/1999', '28/08/1997', '27/09/1997', '28-04-2001', nan, nan, nan, '06/10/1999', '22-01-1999', '18/11/1996',
                                     '27/07/1999', '19-11-2000', '17-09-1981', 'joke', nan, '10-09-1999', '10-1-2000', '19/11/2000', '28-01-1997', '14-05-1999', nan, '02-12-1994',
                                     '13/06/1996', '25-07-2000', '13/12/1996', 'joke', '04-09-1996', '14-05-1999', 'joke']})
    df['When is your birthday (date)?'] = new_column['B']
    df['When is your birthday (date)?'] = df['When is your birthday (date)?'].replace('nan', np.nan)
    df['When is your birthday (date)?'] = df['When is your birthday (date)?'].replace('joke', np.nan)

    #format birthdays
    df["When is your birthday (date)?"] = df["When is your birthday (date)?"].str.replace('-','/')
    df["When is your birthday (date)?"] = df["When is your birthday (date)?"].str.replace('.','/')
    df["When is your birthday (date)?"] = df["When is your birthday (date)?"].str.replace(' ','/')

    #Individual changes
    df.replace('six hundred', '600', inplace=True)
    replacements = {'24:00':'00:00', '12:00': '00:00', '11:00': '23:00', '11:30': '23:30', '11:15': '23:15', '11:45': '23:45'}
    df['Time you went to bed Yesterday'] = df['Time you went to bed Yesterday'].replace(replacements)
    #print(df['Time you went to bed Yesterday'])

    #changing digits and letters to numeric
    numeric_regex = r'-?\d+'
    for index, row in df.iterrows():
        text_value0 = str(row['What is your stress level (0-100)?'])
        text_value1 = str(row['How many students do you estimate there are in the room?'])
        text_value2 = str(row['How many hours per week do you do sports (in whole hours)? '])
        text_value3 = str(row['Give a random number'])
        numeric_value0 = re.search(numeric_regex, text_value0)
        numeric_value1 = re.search(numeric_regex, text_value1)
        numeric_value2 = re.search(numeric_regex, text_value2)
        numeric_value3 = re.search(numeric_regex, text_value3)

        if numeric_value0:
            numeric_value0 = int(numeric_value0.group(0))
            df.at[index, 'What is your stress level (0-100)?'] = numeric_value0
        if numeric_value1:
            numeric_value1 = int(numeric_value1.group(0))
            df.at[index, 'How many students do you estimate there are in the room?'] = numeric_value1
        if numeric_value2:
            numeric_value2 = int(numeric_value2.group(0))
            df.at[index, 'How many hours per week do you do sports (in whole hours)? '] = numeric_value2
        if numeric_value3:
            numeric_value3 = int(numeric_value3.group(0))
            df.at[index, 'Give a random number'] = numeric_value3

    df['What is your stress level (0-100)?'] = pd.to_numeric(df['What is your stress level (0-100)?'], errors='coerce')
    df['How many students do you estimate there are in the room?'] = pd.to_numeric(df['How many students do you estimate there are in the room?'], errors='coerce')
    df['How many hours per week do you do sports (in whole hours)? '] = pd.to_numeric(df['How many hours per week do you do sports (in whole hours)? '], errors='coerce')
    df['Give a random number'] = pd.to_numeric(df['Give a random number'], errors='coerce')

    df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'], format='%d/%m/%Y')

    return df

#######################################################################################################################

#filter birthdates outliers and putting in mean in date format

def outliers(dataframe):
    df = dataframe.copy(deep=True)

    df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'], format='%d/%m/%Y')
    mask = (df['When is your birthday (date)?'] < pd.to_datetime('01/01/1940', format='%d/%m/%Y')) | (
                df['When is your birthday (date)?'] > pd.to_datetime('31/12/2010', format='%d/%m/%Y'))
    df.loc[mask, 'When is your birthday (date)?'] = np.nan


    df.loc[~df['What is your stress level (0-100)?'].between(0, 100), 'What is your stress level (0-100)?'] = np.nan
    df.loc[~df['How many students do you estimate there are in the room?'].between(1,
                                                                                   1000), 'How many students do you estimate there are in the room?'] = np.nan
    df.loc[~df['How many hours per week do you do sports (in whole hours)? '].between(0,
                                                                                      100), 'How many hours per week do you do sports (in whole hours)? '] = np.nan

    print(df['When is your birthday (date)?'])

def conorisanengineer(dataframe):
    df = dataframe.copy(deep=True)

    # Outliers
    df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'], format='%d/%m/%Y')
    mask = (df['When is your birthday (date)?'] < pd.to_datetime('01/01/1940', format='%d/%m/%Y')) | (
            df['When is your birthday (date)?'] > pd.to_datetime('31/12/2010', format='%d/%m/%Y'))
    df.loc[mask, 'When is your birthday (date)?'] = np.nan

    df.loc[~df['What is your stress level (0-100)?'].between(0, 100), 'What is your stress level (0-100)?'] = np.nan
    df.loc[~df['How many students do you estimate there are in the room?'].between(1,
                                                                                   1000), 'How many students do you estimate there are in the room?'] = np.nan
    df.loc[~df['How many hours per week do you do sports (in whole hours)? '].between(0,
                                                                                      100), 'How many hours per week do you do sports (in whole hours)? '] = np.nan
    ########################
    def count_digits(num):
        num_str = '{:.0f}'.format(num)
        num_digits = len(num_str) - (1 if num < 0 else 0) - ('.' in num_str) - ('e' in num_str) - num_str.count('-')
        return num_digits

    df['Give a random number'] = df['Give a random number'].apply(count_digits)

    # birthdays
    today = datetime.today()
    df['age'] = (today - df['When is your birthday (date)?']).dt.days / 365.25

    # bedtimes
    df['Time you went to bed Yesterday'] = pd.to_datetime(df['Time you went to bed Yesterday'], format='%H:%M')

    # Create a reference date at midnight
    ref_date = pd.to_datetime('00:00', format='%H:%M')

    # Calculate the time difference in hours between the reference date and each time
    df['time_diff'] = (df['Time you went to bed Yesterday'] - ref_date).dt.seconds / 3600

    # Adjust the times to be centered around midnight
    df['centered_bedtime'] = df['time_diff'] - 24
    df.loc[df['centered_bedtime'] < -12, 'centered_bedtime'] += 24
    df.loc[(df['centered_bedtime'] >= -12) & (df['centered_bedtime'] <= -6), 'centered_bedtime'] += 12

    df['hours of sleep'] = 9 - df['centered_bedtime']  # assume wake up at 9am
    df = df.drop('time_diff', axis=1)

    ''''
    ################### replacing unknowns with most frequent in catagorical
    mf0 = df['Have you taken a course on machine learning?'].value_counts().idxmax()
    df['Have you taken a course on machine learning?'] = df['Have you taken a course on machine learning?'].replace('unknown', mf0)

    mf1 = df['Have you taken a course on information retrieval?'].value_counts().idxmax()
    df['Have you taken a course on information retrieval?'] = df['Have you taken a course on information retrieval?'].replace(
        'unknown', mf1)

    mf2 = df['Have you taken a course on statistics?'].value_counts().idxmax()
    df['Have you taken a course on statistics?'] = df['Have you taken a course on statistics?'].replace(
        'unknown', mf2)

    mf3 = df['Have you taken a course on databases?'].value_counts().idxmax()
    df['Have you taken a course on databases?'] = df['Have you taken a course on databases?'].replace(
        'unknown', mf3)

    mf4 = df['I have used ChatGPT to help me with some of my study assignments '].value_counts().idxmax()
    df['I have used ChatGPT to help me with some of my study assignments '] = df['I have used ChatGPT to help me with some of my study assignments '].replace(
        'not willing to say', mf4)

    mf5 = df['Did you stand up to come to your previous answer    ?'].value_counts().idxmax()
    df['Did you stand up to come to your previous answer    ?'] = df['Did you stand up to come to your previous answer    ?'].replace(
        'unknown', mf5)
    '''

    #replace missing randos and ages with median and people who unknown standup = no

    mean_rando = df['Give a random number'].median(skipna=True)
    df['Give a random number'].fillna(mean_rando, inplace=True)

    mean_age = df['age'].median(skipna=True)
    df['age'].fillna(mean_age, inplace=True)

    mf5 = df['Did you stand up to come to your previous answer    ?'].value_counts().idxmax()
    df['Did you stand up to come to your previous answer    ?'] = df[
        'Did you stand up to come to your previous answer    ?'].replace(
        'unknown', mf5)

    ###################preparedness thing

    preparedness_df = pd.DataFrame()
    preparedness_df['Have you taken a course on machine learning?'] = df['Have you taken a course on machine learning?'].replace(
        {'yes': 1, 'no': 0, 'unknown': 0})
    preparedness_df['Have you taken a course on information retrieval?'] = df[
        'Have you taken a course on information retrieval?'].replace({'unknown': 0})
    preparedness_df['Have you taken a course on statistics?'] = df['Have you taken a course on statistics?'].replace(
        {'mu': 1, 'sigma': 0, 'unknown': 0})
    preparedness_df['Have you taken a course on databases?'] = df['Have you taken a course on databases?'].replace(
        {'ja': 1, 'nee': 0, 'unknown': 0})
    preparedness_df['I have used ChatGPT to help me with some of my study assignments '] = df['I have used ChatGPT to help me with some of my study assignments '].replace(
        {'ja': 1, 'nee': 0, 'not willing to say': 0})


    preparedness_df['preparedness'] = preparedness_df.sum(axis=1)
    #print(preparedness_df['preparedness'])


    #KNN stuff

    # change some categorical to numerical first for more 'training' data
    df['Have you taken a course on machine learning?'] = df['Have you taken a course on machine learning?'].replace({'yes': 1, 'no': 0, 'unknown': np.nan})
    df['Have you taken a course on information retrieval?'] = df['Have you taken a course on information retrieval?'].replace({'unknown': np.nan})
    df['Have you taken a course on statistics?'] = df['Have you taken a course on statistics?'].replace(
        {'mu': 1, 'sigma': 0, 'unknown': np.nan})
    df['Have you taken a course on databases?'] = df['Have you taken a course on databases?'].replace(
        {'ja': 1, 'nee': 0, 'unknown': np.nan})
    df['I have used ChatGPT to help me with some of my study assignments '] = df['I have used ChatGPT to help me with some of my study assignments '].replace(
        {'yes': 1, 'no': 0, 'not willing to say': np.nan})
    df.loc[~df['What is your gender?'].isin(['male', 'female']), 'What is your gender?'] = '?'
    df['What is your gender?'] = df['What is your gender?'].replace(
        {'?': -1, 'male': 0, 'female': 1})
    df['Did you stand up to come to your previous answer    ?'] = df['Did you stand up to come to your previous answer    ?'].replace(
        {'yes': 1, 'no': 0, 'unknown': np.nan})

    #still need to add programme in numerical
    cat_df = df[['Have you taken a course on machine learning?','Have you taken a course on information retrieval?',
                       'Have you taken a course on statistics?','Have you taken a course on databases?',
                 'What is your gender?', 'I have used ChatGPT to help me with some of my study assignments ']]
    imputer = KNNImputer(n_neighbors=20)
    imputer.fit_transform(cat_df.sample(frac=2/3, random_state = 57)) #now its trained
    imputed_cat_df = pd.DataFrame(imputer.fit_transform(cat_df), columns = cat_df.columns)
    imputed_cat_df = imputed_cat_df.applymap(lambda x: round(x * 2) / 2)
    #print(imputed_cat_df['Have you taken a course on information retrieval?'])


    howmany_df = df[['Have you taken a course on machine learning?','Have you taken a course on information retrieval?',
                       'Have you taken a course on statistics?','Have you taken a course on databases?',
                      'Did you stand up to come to your previous answer    ?','How many students do you estimate there are in the room?']]
    imputer = KNNImputer(n_neighbors=20)
    imputer.fit_transform(howmany_df.sample(frac=2 / 3, random_state=57))  # now its trained
    imputed_howmany_df = pd.DataFrame(imputer.fit_transform(howmany_df), columns = howmany_df.columns)

    rest_df = df[
        ['Have you taken a course on machine learning?', 'Have you taken a course on information retrieval?',
         'Have you taken a course on statistics?', 'Have you taken a course on databases?',
         'What is your stress level (0-100)?','How many hours per week do you do sports (in whole hours)? ' , 'hours of sleep']]
    imputer = KNNImputer(n_neighbors=20)
    imputer.fit_transform(rest_df.sample(frac=2 / 3, random_state=57))  # now its trained
    imputed_rest_df = pd.DataFrame(imputer.fit_transform(rest_df), columns=rest_df.columns)


    '''
    # Fill with mean
    
    mean_bday= df['When is your birthday (date)?'].mean(skipna=True)
    df['When is your birthday (date)?'].fillna(mean_bday, inplace=True)
    
    mean_stress = df['What is your stress level (0-100)?'].mean(skipna=True)
    df['What is your stress level (0-100)?'].fillna(mean_stress, inplace=True)

    mean_sit = df['How many students do you estimate there are in the room?'].mean(skipna=True)
    df['How many students do you estimate there are in the room?'].fillna(mean_sit, inplace=True)

    mean_sports = df['How many hours per week do you do sports (in whole hours)? '].mean(skipna=True)
    df['How many hours per week do you do sports (in whole hours)? '].fillna(mean_sports, inplace=True)
    '''

    '''
    df['age'] = imputed_df['age']
    df['How many students do you estimate there are in the room?'] = imputed_df['How many students do you estimate there are in the room?']
    df['What is your stress level (0-100)?'] = imputed_df['What is your stress level (0-100)?']
    df['How many hours per week do you do sports (in whole hours)? '] = imputed_df['How many hours per week do you do sports (in whole hours)? ']
    df['hours of sleep'] = imputed_df['hours of sleep']
    df['preparedness'] = preparedness_df['preparedness']
    #df['centered_bedtime'] = imputed_df['centered_bedtime']
    '''

    #now bring together and remove rows

    new_df = pd.DataFrame()

    new_df['What programme are you in?'] = df['What programme are you in?'] # this needs to be in 0-11
    new_df['What makes a good day for you (1)?'] = df['What makes a good day for you (1)?']
    new_df['What makes a good day for you (2)?'] = df['What makes a good day for you (2)?']

    new_df['What is your gender?'] = imputed_cat_df['What is your gender?']
    new_df['How many students do you estimate there are in the room?'] = imputed_howmany_df['How many students do you estimate there are in the room?']
    new_df['hours of sleep'] = imputed_rest_df[
        'hours of sleep']
    new_df['How many hours per week do you do sports (in whole hours)? '] = imputed_rest_df[
        'How many hours per week do you do sports (in whole hours)? ']
    new_df['What is your stress level (0-100)?'] = imputed_rest_df[
        'What is your stress level (0-100)?']

    new_df['preparedness'] = preparedness_df['preparedness']
    new_df['Give a random number'] = df['Give a random number']
    new_df['age'] = df['age']
    new_df['Did you stand up to come to your previous answer    ?'] = df['Did you stand up to come to your previous answer    ?']

    return new_df

#######################################################################################################################

dataframe = pd.read_csv('ODI-2023.csv', sep=';')
dataframe = magdas_bedtime_cleaning(dataframe)
dataframe = conorclean(dataframe)
dataframe.to_excel('Conor_cleaned.xlsx', index=False)
#dateframe = outliers(dataframe)
dataframe = conorisanengineer((dataframe))
#print(dataframe)

#print(len(dataframe['Time you went to bed Yesterday'].dropna()))
#print(min(dataframe['centered_bedtime'].dropna()))

dataframe.to_excel('Conor_cleaned_engeneered.xlsx', index=False)


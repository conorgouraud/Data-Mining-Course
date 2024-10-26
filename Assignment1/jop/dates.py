import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# df = pd.read_excel('ODI-2023.xlsx')
def clean_dates_numerics(dataframe):
    df = dataframe.copy(deep=True)
    nan = 'nan'
    new_column = pd.DataFrame({'B': ['23/11/1997', '08/05/1995', nan, '25-06-1996', '18-08-1998', '13-03-2000', '21-07-1999', '08-03-2000', '24-11-1996', '07.12.1998', '02-01-1999', '14/11/1999', 'joke', nan, nan, '25.09.2000', '04 02 1999', '27-11-1999', '29-12-1997', '09-02-2000', '09-05-1999', '14-05-1999', '11/11/1997', nan, '08-08-1997', '11-05-1998', '15-09-2002', '26/04/1997', '24-10-2000', '05.07.2002', '31-01-2000', '03/05/1997', '31-03-1999', nan, '18-12-1998', '17/02/1998', '22/08/1999', '28-09-1998', '5 07 2001', '12-03-1999', '14-12-1997', '31/10/2001', '10/01/2001', '01/25/1998', '24/04/1996', '08-08-1999', '18-06-1999', '19/09/1998', '08-09-2000', '17-04-1998', '02-11-2000', '06-09-1998', '28-2-2000', '30 03 1998', '12-06-1998', '16-01-1999', '22.04.1996', '12-08-1999', nan, '2/6/1997', '28-05-1999', '28 12 2000', '31-03-1998', '02-05-1999', '06/11/2000', '02-06-2000', nan, '19-02-2000', '07/08/1997', '01/05/1997', '03/31/1991', '08-07-2001', '17/6/1997', nan, '18-08-1996', nan, '04-12-1998', '16-09-1999', nan, '5/11/2000', '26/10/1996', nan, '23-10-2001', '22 06 1999', nan, '12-08-2000', '21/02/2000', '15-09-1999', '12.12.1997', '04.09.2000', 'joke', '23-08-1999', '31-08-1997', '23-05-2000', '24/11/1999', nan, nan, '27.09.1999', '02-05-2000', nan, '03/04/1998', 'joke', '10-09-1997', nan, '08-06-2000', '12.02.1997', nan, '30/04/1998', '15/07/1997', '17.04.1997', '14-06-1999', '30 05 1997', '6-8-1997', '06/05/1999', '04/03/2000', 'joke', 'joke', '11-08-1997', '6/11/1999', nan, '19-05-2000', '14-08-2000', '5-2-1997', '14/07/1994', '23/4/1985', '30 09 1997', '22-12-1997', '13/03/1999', '11-05-1998', '25/04/1999', '25-05-1999', '05/12/1999', '20.01.1999', nan, '15-01-2001', nan, '11/08/1992', '03 07 1998', '25-04-1987', '08-08-2001', '17/01/1997', nan, nan, '15-11-1998', 'joke', '10/03/2001', nan, '15-01-1999', '07-10-1999', '07-07-1995', nan, '11-09-2000', '09-07-2000', '04/11/2000', '10-01-2001', '21-12-1999', '23.12.1999', '06-11-1999', '01/01/1990', '02.08.1998', '03-05-2000', '16/06/1999', '14-12-2000', '21/03/2000', '23-03-1998', nan, '23/11/1997', '18-11-2000', '08/05/1998', '05 01 1998', '09/03/1998', 'joke', '29/11/2000', nan, '31/03/1998', '10-05-2001', nan, '26/06/1999', '01/05/1995', '20/06/1999', nan, '10-10-2001', '18-08-1999', '29-06-2000', '11-09-1999', '18-11-2000', '21-09-1999', nan, '20/11/1990', 'joke', '23.04.1995', '12-08-1999', nan, nan, '12/03/1999', '24-08-1999', '14-09-2001', '19-05-2000', nan, '20/09/1998', '28/10/1996', '25-11-2000', '05-05-1998', '05/11/1999', '01/06/1999', '23-08-1999', '05/02/1998', 'joke', '13-06-2000', '17-04-1998', '06/11/1997', '15-4-1999', '21-02-2000', nan, '09-09-2001', '09/10/2000', 'joke', '20.06.2000', '07-07-2001', nan, '04/02/1999', '11/03/1995', nan, '23-02-1999', nan, '19 09 1997', '30/12/2000', '04-01-2001', '15/09/1999', '11-12-1997', '14/05/1998', '05-09-2000', '27-11-2000', '30/12/1999', '10/12/1998', '29-05-2000', 'joke', '03-01-2001', '19-11-1999', '20-02-2002', '16/01/2001', nan, '4/6/2000', 'joke', '06-02-1996', nan, '04-08-1998', nan, 'joke', '29 03 1998', 'joke', nan, '18.08.1997', 'joke', 'joke', nan, '31-12-1999', '27-10-1997', '13/07/1999', '08/08/1998', '09/09/1993', '06-29-2000', '26-10-2000', '22 11 1999', 'joke', '24-01-2000', '14-09-2000', nan, 'joke', '12-02-2001', '28 09 2000', nan, '15-01-1999', nan, '6-11-1997', '10-07-2000', '02/11/1999', '28/08/1997', '27/09/1997', '28-04-2001', nan, nan, nan, '06/10/1999', '22-01-1999', '18/11/1996', '27/07/1999', '19-11-2000', '17-09-1981', 'joke', nan, '10-09-1999', '10-1-2000', '19/11/2000', '01-28-1997', '14-05-1999', nan, '02-12-1994', '13/06/1996', '25-07-2000', '13/12/1996', 'joke', '04-09-1996', '14-05-1999', 'joke']})
    df['When is your birthday (date)?'] = new_column['B']
    df['When is your birthday (date)?'] = df['When is your birthday (date)?'].replace('nan', np.nan)
    df['When is your birthday (date)?'] = df['When is your birthday (date)?'].replace('joke', np.nan)

    #format birthdays
    df["When is your birthday (date)?"] = df["When is your birthday (date)?"].str.replace('-','/')
    df["When is your birthday (date)?"] = df["When is your birthday (date)?"].str.replace('.','/')
    df["When is your birthday (date)?"] = df["When is your birthday (date)?"].str.replace(' ','/')

    for i in df["When is your birthday (date)?"]:
        j = str(i)
        j = j.split('/')

        try:
            j = [int(x) for x in j]
            n = j[2]+((j[1]-1)*30.5+j[0]-1)/365.25
            df["When is your birthday (date)?"].replace(i, n , inplace=True)
            # df.at[i, 'How many students do you estimate there are in the room?'] = n
        except ValueError:
            pass

    #print(df['When is your birthday (date)?'])
    #for i in df['When is your birthday (date)?']:
        #print(i)

    bdays = list(df['When is your birthday (date)?'])
    plt.hist(bdays, 50)
    #plt.show()

    #Individual changes
    df.replace('six hundred', '600', inplace=True)


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

    #print(df)

    df['What is your stress level (0-100)?'] = pd.to_numeric(df['What is your stress level (0-100)?'], errors='coerce')
    df['How many students do you estimate there are in the room?'] = pd.to_numeric(df['How many students do you estimate there are in the room?'], errors='coerce')
    df['How many hours per week do you do sports (in whole hours)? '] = pd.to_numeric(df['How many hours per week do you do sports (in whole hours)? '], errors='coerce')
    df['Give a random number'] = pd.to_numeric(df['Give a random number'], errors='coerce')

    #print(df)
    return df
    df.to_excel('Conor_cleaned.xlsx', index=False)

#######################################################################################################################
if __name__ == "__main__":
    df = pd.read_excel('ODI-2023.xlsx')
    df['How many students do you estimate there are in the room?'] = pd.to_numeric(df['How many students do you estimate there are in the room?'], errors='coerce')
    studentguess = df['How many students do you estimate there are in the room?']
    studentguess = studentguess.dropna()

    #print(studentguess)

    #plt.hist(studentguess, 100)
    #plt.show()

    #idk_df = df[df['Did you stand up to come to your previous answer    ?'] == "unknown"]
    #print(np.mean(idk_df['How many students do you estimate there are in the room?'].dropna()))

    standup = df[df['Did you stand up to come to your previous answer    ?'] != 'unknown']
    yes_df = standup[standup['Did you stand up to come to your previous answer    ?'] == "yes"]
    print(np.mean(yes_df['How many students do you estimate there are in the room?'].dropna())) # average guess of people who stood
    no_df = standup[standup['Did you stand up to come to your previous answer    ?'] == "no"]
    print(np.mean(no_df['How many students do you estimate there are in the room?'].dropna()))

    #######################################################################################################################

    randoms = df['Give a random number'].dropna()
    randoms = pd.to_numeric(randoms, errors='coerce')
    index_to_remove = randoms.nlargest(50).index
    #print(index_to_remove)
    #randoms = randoms.drop(index_to_remove)

    #print(np.median(randoms))

    #plt.hist(randoms, 100)
    #plt.show()

    #######################################################################################################################

    stress = df[df['What is your stress level (0-100)?'] != 'joke']
    stress = stress['What is your stress level (0-100)?']
    stress = stress.dropna()
    stress = pd.to_numeric(stress, errors='coerce')
    #print(len(stress))

    #######################################################################################################################

    sports = df[df['How many hours per week do you do sports (in whole hours)? '] != 'joke']
    sports = sports['How many hours per week do you do sports (in whole hours)? ']
    sports = sports.dropna()
    sports = pd.to_numeric(sports, errors='coerce')
    #print(np.median(sports))

    #######################################################################################################################

    ML = df['Did you stand up to come to your previous answer    ?']
    count = ML.value_counts()
    #print(count)
    #count.plot(kind='pie')
    #plt.show()

    #######################################################################################################################
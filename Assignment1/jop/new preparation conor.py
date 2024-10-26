from cleaning import bedtime_cleaning, date_cleaning, misc_cleaning
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


def reverse_dict(dictionary):
    '''
    Reverse input dictionary by switching keys and values.

    Arguments:
        dictionary:
            the key is a string that must become the value
            the value is a list of strings that must all become keys with their repsective key as the value
    '''

    reverse_dict = {}
    for key, values in dictionary.items():
        for word in values:
            reverse_dict[word] = key
    return reverse_dict


def remove_redundant_words(text, redundant_words):
    '''
    Function to simplify an input string by removing all occurences of predefined substrings.

    Arguments:
        text:               input string to simplify
        redundant_words:    list of strings to remove from input text

    Returns:
        simplified version of the input string
    '''
    # Perform simple string operations on input text
    word = str(text).lower()
    word = word.replace('.', '')
    word = word.replace(',', '')

    word = word.replace('-', '')
    word = word.replace('“', '')

    word = char_cutoff(word, {':', '('})
    word = word.lower()
    word = word.replace('&', 'and')
    for redundant_word in redundant_words:
        word = word.replace(redundant_word, '')
    return word.strip()


def char_cutoff(text, chars):
    '''
    Function that simplifies a string by cutting off everything after the first occurence of any character from a supplied set of characters.

    Arguments:
        text:   input string to simplify
        chars:  set of characters signifying cutoff points

    Returns:
        Simplified version of the input string
    '''
    # Replace all cutoff characters with the same sequence of characters and split on that sequence
    for char in chars:
        text = text.replace(char, '??')
    split_text = text.split('??')
    return split_text[0].strip()


def regroup(input_value, series, threshold, alt_value):
    amount = len(series.loc[series == input_value])
    if amount < threshold:
        return alt_value
    else:
        return input_value


def categorize_spelling(text, spellings, redundant_words):
    '''
    Function that assigns an input word to a category based on a provided dictionary of spellings, e.g. 'artificial intelligence'
    and 'ai' both belong to category 'Artificial Intelligence'.

    Arguments
        text:               string to categorize
        spellings:          dictionary:
                                the key is a string representing a valid spelling.
                                the value is a string that represents the associated category
        redundant_words:    list of strings to filter out of the input text prior to categorizing

    Returns:
        category:           string representing the determined category
    '''

    # # Remove words from a predefined set of unwanted words from the input text
    # word = remove_redundant_words(word, redundant_words)

    # word = word.strip()
    # Assign category based on provided dictionary if possible, else on the updated input text
    category = spellings.get(text, text)
    return category


def categorize_keywords(text, keywords):
    '''
    Function that splits an input string into words and checks those words for an associated category.

    Arguments
        text:       string to categorize
        keywords:   dictionary:
                        the key is a string representing a valid keyword.
                        the value is a string that represents the associated category
    Returns:
        category:   string representing the determined category
    '''
    words = text.split()
    # Try to find associated category based on words in the input text
    for word in words:
        try:
            category = keywords[word]
            break
        except KeyError:
            category = np.nan
    return category


def clean_numerics(input_value, lower_bound, upper_bound, error_value):
    try:
        if lower_bound <= input_value <= upper_bound:
            output_value = input_value
        else:
            output_value = error_value

    except ValueError:
        output_value = np.nan
    return output_value


def length(input_value):
    return len(str(input_value))


if __name__ == '__main__':

    with open("supplementary_data.json", "r") as info:
        suppl_data = json.load(info)

    questions = suppl_data['questions']
    program_spellings = suppl_data['program_spellings']
    program_keywords = suppl_data['program_keywords']

    df = pd.read_excel('ODI-2023.xlsx')

    for i, cleaning_function in enumerate([bedtime_cleaning, date_cleaning, misc_cleaning]):
        df = cleaning_function(df)

    redundant_words = suppl_data['redundant_words']
    df[questions['master_program']] = df[questions['master_program']].apply(remove_redundant_words,
                                                                            args=(redundant_words,))

    reverse_program_spellings = reverse_dict(program_spellings)
    reverse_program_keywords = reverse_dict(program_keywords)
    column = questions['master_program']

    # Simplify program column by accounting for different spellings/phrasing up of equal MSc programs
    df[column] = df[column].apply(categorize_spelling, args=(reverse_program_spellings, redundant_words))
    leftovers = df.loc[~df[column].isin(program_spellings.keys()), column]

    # Simplify remaining MSc program inputs based on keywords
    leftovers = leftovers.apply(categorize_keywords, args=(reverse_program_keywords,))
    df.loc[~df[questions['master_program']].isin(program_spellings.keys()), column] = leftovers

    leftovers = df.loc[~df[questions['master_program']].isin(program_spellings.keys()), column]
    df[column] = df[column].apply(regroup, args=(df[column].copy(), 5, 'Other'))

    ################################################################################
    # From here the msc programmes are cleaned and regrouped into ' Other', when <= 5 occurences,
    # so if you impute the missing programmes (np.nan) here it should work

    ################################################################################
    # End of Conor's engineering zone
    programs = df[column]

    df[questions["good_day_1"]] = df[questions["good_day_1"]].apply(length)
    df[questions["good_day_2"]] = df[questions["good_day_2"]].apply(length)
    #print(df[[questions["good_day_1"], questions["good_day_2"]]][:10].to_markdown())
    df[questions["good_day_1"]] = df[[questions["good_day_1"], questions["good_day_2"]]].sum(axis=1)

    df.drop(questions["good_day_2"], axis=1, inplace=True)

    df.to_excel('cleaned_dataset.xlsx', index=False)

    #####Q4

    #onehot encode programs and genders
    ohe = OneHotEncoder(sparse=False)
    pro_encoded = ohe.fit_transform(df[['What programme are you in?']])
    pro_encoded_df = pd.DataFrame(pro_encoded, columns=['a', 'b','c','d','e','f','g', 'h', 'i'])
    #df = pd.concat([df, pro_encoded_df], axis=1)

    gender_encoded = ohe.fit_transform(df[['What is your gender?']])
    gender_encoded_df = pd.DataFrame(gender_encoded, columns=['x', 'y','z' ])
    #df = pd.concat([df, gender_encoded_df], axis=1)

    #drop columns not using
    df.drop("What is your gender?", axis=1, inplace=True)
    df.drop("What programme are you in?", axis=1, inplace=True)
    df.drop("What makes a good day for you (1)?", axis=1, inplace=True)
    #df.drop("Did you stand up to come to your previous answer    ?", axis=1, inplace=True)
    #df.drop("How many students do you estimate there are in the room?", axis=1, inplace=True)
    df.drop("Give a random number", axis=1, inplace=True)
    #df.drop("hours of sleep", axis=1, inplace=True)
    #df.drop("How many hours per week do you do sports (in whole hours)? ", axis=1, inplace=True)
    #df.drop("preparedness", axis=1, inplace=True)
    df.drop("age", axis=1, inplace=True)

    y = df["What is your stress level (0-100)?"]

    df.to_excel('regression_data.xlsx', index=False)

    X = df.drop("What is your stress level (0-100)?", axis=1)

    test_sizes = np.linspace(0.2,0.6,15)
    #test_sizes = [0.33]
    r2_list = []
    for i in test_sizes:
        lr = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=57)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        r2_list.append(r2)
        print('R2 score:', r2)
    plt.plot(test_sizes, r2_list)
    plt.xlabel('Test set size')
    plt.ylabel('$R^2$ value ')
    #plt.show()

    # Results for the optimal test size (0.33)

    lr = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=57)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    print('Slopes:', lr.coef_)
    print('Intercept:', lr.intercept_)

    sports = df['How many hours per week do you do sports (in whole hours)? ']
    sports_test = X_test['How many hours per week do you do sports (in whole hours)? ']
    print(sports_test)
    plt.scatter(sports, y, color='blue')
    plt.plot(sports_test, y_pred, color='red')
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.title('Linear Regression')
    plt.show()

    mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
    print('Mean Absolute Error is:', mae)

    mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
    print('Mean Square Error is:', mse)
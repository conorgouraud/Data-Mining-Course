from bedtimes import clean_times, plot_times
from dates import clean_dates_numerics
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def regroup(df, input_value, threshold, alt_value):
    amount = len(column.loc[column==input])
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
            category = f'Unknown'
    return category


def clean_numerics(input_value, lower_bound, upper_bound, error_value):
    try:
        if lower_bound <= input_value <= upper_bound:
            output_value = input_value
        else:
            output_value = error_value

    except ValueError:
        output_value = error_value
    return output_value


def pie_chart(column, labels):
    # print(column)
    amounts = []
    labels = list(labels)
    
    misc_amount = 0
    remove_labels = []
    for label in labels:
        amount = len(column.loc[column==label])
        if amount < 5:
            misc_amount += amount
            remove_labels.append(label)
        else:
            amounts.append(len(column.loc[column==label]))
    labels.append('Misc.')
    amounts.append(misc_amount)
    labels = [label for label in labels if label not in remove_labels]
    labels.append('Unknown')
    amounts.append(len(column) - sum(amounts))
    plt.pie(amounts, labels=labels)
    plt.show()

if __name__ == '__main__':
    with open("supplementary_data.json", "r") as info:

        suppl_data = json.load(info)
    
    questions = suppl_data['questions']
    print(questions)
    # raise NotImplementedError
    # questions = {1: 'Tijdstempel', 
    #               2: 'What programme are you in?', 
    #               3: 'Have you taken a course on machine learning?', 
    #               4: 'Have you taken a course on information retrieval?', 
    #               5: 'Have you taken a course on statistics?', 
    #               6: 'Have you taken a course on databases?', 
    #               7: 'What is your gender?', 
    #               8: 'I have used ChatGPT to help me with some of my study assignments ' , 
    #               9: 'When is your birthday (date)?', 
    #               10: 'How many students do you estimate there are in the room?', 
    #               11: 'Did you stand up to come to your previous answer    ?', 
    #               12: 'What is your stress level (0-100)?', 
    #               13: 'How many hours per week do you do sports (in whole hours)?' , 
    #               14: 'Give a random number', 
    #               15: 'Time you went to bed Yesterday', 
    #               16: 'What makes a good day for you (1)?', 
    #               17: 'What makes a good day for you (2)?'}
    
    df = pd.read_excel('ODI-2023.xlsx')
    # print(df[questions['bed_time']].to_markdown())
    df[questions['bed_time']] = clean_times(df[questions['bed_time']])

    plot_times(df[questions['bed_time']])
    df = clean_dates_numerics(df)
    # print(df[questions['bed_time']].to_markdown())
    # for key in ['machine_learning', 'information_retrieval', 'statistics', 'databases', 'chatgpt', 'stand_up']:
    #     df[questions[key]] = df[questions[key]].apply({'yes': 'Yes', 'no': 'No', '1': 'Yes', '0': 'No', 'mu': 'Yes', 'sigma': 'No', 'ja': 'Yes', 'nee': 'No'})
    #     print(df[questions[key]].to_markdown())

    redundant_words = ['bachelor', 'msc', 'mc ', 'mcs', 'masters ', 'master ',' master', ' ms', 'master\'s', 'master of']
    df[questions['master_program']] = df[questions['master_program']].apply(remove_redundant_words, args=(redundant_words,))
    programme_spellings = {'Artificial Intelligence': ['ai', 'artificial intelligence'],
                           'Business Analytics': ['business analytics', 'ba'],
                           'Bioinformatics and Systems Biology': ['bioinformatics', 'bioinformatics and systemsbiology', 'bioinformatics and system biology',
                                                                  'bioinformatics and systems biology', 'bioinformatics & systems biology'],
                           'Biomedical Sciences': ['biomedical sciences'],
                           'Computational Science': ['cls', 'computational science'],
                           'Digital Business and Innovation': ['digital business and innovation'], 
                           'Computer Science': ['cs', 'computer science', 'big data engineering'],
                           'Economics': ['economics'],
                           'Econometrics': ['econometrics', 'econometrics and data science', 'econometrics and datascience', 'data science and econometrics', 'econometrics and operations research'],
                           'Finance': ['finance and technology', 'finance'],
                           'Human Language Technology': ['human language technology'],
                           'Information Sciences': ['information sciences'],
                           'Management, Policy Analysis and Entrepreneurship in Health and Life Sciences': ['management policy analysis and entrepreneurship in health and life sciences'],
                           'Neurosciences': ['neuroscience'],
                           'Quantitative Risk Management': ['quantitative risk management', 'qrm'],
                           'Stochastics and Financial Mathematics': ['stochastics and financial mathematics']}

    programme_keywords = {'Artificial Intelligence': ['ai', 'artificial', 'artifscial', 'intelligence'],
                          'Bioinformatics and Systems Biology': ['bio-informatics', 'bioinformatics', 'bionformatics'],
                          'Computer Science': ['computer', 'cs'], 
                          'Econometrics': ['econometrics'],
                          'Economics': ['economics'],
                          'Finance': ['finance', 'fintech', 'duisenberg'],
                          'Quantitative Risk Management': ['quantitative']}

    reverse_programme_spellings = reverse_dict(programme_spellings)
    reverse_programme_keywords = reverse_dict(programme_keywords)
    column = questions['master_program']

    # Simplify programme column by accounting for different spellings/phrasing up of equal MSc programmes
    df[column] = df[column].apply(categorize_spelling, args=(reverse_programme_spellings, redundant_words))
    leftovers = df[column].loc[~df[questions['master_program']].isin(programme_spellings.keys())]

    # Simplify remaining MSc programme inputs based on keywords 
    leftovers = leftovers.apply(categorize_keywords, args=(reverse_programme_keywords,))
    df[column].loc[~df[questions['master_program']].isin(programme_spellings.keys())] = leftovers

    leftovers = df[column].loc[~df[questions['master_program']].isin(programme_spellings.keys())]

    # print(len(leftovers))
    # print(df[column].to_markdown())

    # print(leftovers.to_markdown())
    # print(len(leftovers))

    programmes = df[column]
    stress = questions['stress_level']
    df[stress] = df[stress].apply(clean_numerics, args=(0, 100, -1))
    valid_stress_levels = df[stress].loc[df[stress] != -1]
    mean = valid_stress_levels.mean()
    median = valid_stress_levels.median()
    df[stress].loc[df[stress] == -1] = mean

    print(df[stress].to_markdown())
    df.to_excel('cleaned_dataset.xlsx', index=False)
    pie_chart(programmes, programme_spellings.keys())
    # print(df[questions[2]].loc[~df[questions[2]].isin(programme_spellings.keys())])
    # print(df)
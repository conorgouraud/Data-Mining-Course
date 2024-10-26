import matplotlib.pyplot as plt
import numpy as np
from ordered_set import OrderedSet
import pandas as pd


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

programs = list(programme_spellings.keys())

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


def bar_plot(dataframe, main_column_name, sub_column_name):
    main_column = dataframe[main_column_name]
    sub_column = dataframe[sub_column_name]

    main_category_counts = pd.Index(main_column).value_counts()
    sub_category_counts = pd.Index(sub_column).value_counts()

    main_categories = list(set(main_column))
    sub_categories = list(set(sub_column))
    main_categories.sort(key=lambda category: main_category_counts[category], reverse=True)
    sub_categories.sort(key=lambda category: sub_category_counts[category], reverse=True)
    
    data = {key:[] for key in sub_categories}
    # print(pd.Index(main_column))
    # print(main_categories)
    # print(sub_categories)
    # return
    for main_category in main_categories:
        # print(main_category)
        for sub_category in sub_categories:
            # print(sub_category)
            series = dataframe.loc[(main_column == main_category) & (sub_column == sub_category)]
            data[sub_category].append(len(series))


    # print(sub_categories)
    # print(sub_category_counts)

    df = pd.DataFrame(data, index=main_categories)
    ax = df.plot.bar(rot=25, stacked=True)
    plt.title(f'{sub_column_name}')
    plt.savefig(f'images/{sub_column_name.strip("? ")}.png', dpi=300)
    # plt.show()
    # print(main_category_counts)
    # categories.sort(key=lambda category: main_column_counts[category], reverse=True)

    # print([main_column_counts[category] for category in categories])
    return

if __name__ == "__main__":
    questions = {1: 'Tijdstempel', 
                  2: 'What programme are you in?', 
                  3: 'Have you taken a course on machine learning?', 
                  4: 'Have you taken a course on information retrieval?', 
                  5: 'Have you taken a course on statistics?', 
                  6: 'Have you taken a course on databases?', 
                  7: 'What is your gender?', 
                  8: 'I have used ChatGPT to help me with some of my study assignments ' , 
                  9: 'When is your birthday (date)?', 
                  10: 'How many students do you estimate there are in the room?', 
                  11: 'Did you stand up to come to your previous answer    ?', 
                  12: 'What is your stress level (0-100)?', 
                  13: 'How many hours per week do you do sports (in whole hours)?' , 
                  14: 'Give a random number', 
                  15: 'Time you went to bed Yesterday', 
                  16: 'What makes a good day for you (1)?', 
                  17: 'What makes a good day for you (2)?'}
    
    programme_spellings = {'Artificial Intelligence': ['ai', 'artificial intelligence'],
                           'Business Analytics': ['business analytics', 'ba'],
                           'Bioinformatics and Systems Biology': ['bioinformatics', 'bioinformatics and systemsbiology', 
                                                                  'bioinformatics and system biology', 'bioinformatics and systems biology',
                                                                  'bioinformatics & systems biology'],
                           'Biomedical Sciences': ['biomedical sciences'],
                           'Computational Science': ['cls', 'computational science'],
                           'Digital Business and Innovation': ['digital business and innovation'], 
                           'Computer Science': ['cs', 'computer science', 'big data engineering'],
                           'Economics': ['economics'],
                           'Econometrics': ['econometrics', 'econometrics and data science', 'econometrics and datascience',
                                            'data science and econometrics', 'econometrics and operations research'],
                           'Finance': ['finance and technology', 'finance'],
                           'Human Language Technology': ['human language technology'],
                           'Information Sciences': ['information sciences'],
                           'Management, Policy Analysis and Entrepreneurship in Health and Life Sciences': 
                                ['management policy analysis and entrepreneurship in health and life sciences'],
                           'Neurosciences': ['neuroscience'],
                           'Quantitative Risk Management': ['quantitative risk management', 'qrm'],
                           'Stochastics and Financial Mathematics': ['stochastics and financial mathematics']}
    
    programs = list(programme_spellings.keys())
    df = pd.read_excel('cleaner_dataset.xlsx')
    # print(df)
    # print(df[categories[2]].to_markdown())
    # print(set(df[categories[2]]))
    for key in [3, 4, 5, 6, 7, 8]:
        bar_plot(df, questions[2], questions[key])
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def bar_plot(dataframe, main_column_name, sub_column_name, labels_dict):
    
    main_column = dataframe[main_column_name]
    main_column = main_column.map(labels_dict)
    sub_column = dataframe[sub_column_name]

    main_category_counts = pd.Index(main_column).value_counts()
    sub_category_counts = pd.Index(sub_column).value_counts()

    main_categories = list(set(main_column))
    sub_categories = list(set(sub_column))
    main_categories.sort(key=lambda category: main_category_counts[category], reverse=True)
    sub_categories.sort(key=lambda category: sub_category_counts[category], reverse=True)
    
    data = {key:[] for key in sub_categories}

    for main_category in main_categories:
        for sub_category in sub_categories:
            series = dataframe.loc[(main_column == main_category) & (sub_column == sub_category)]
            data[sub_category].append(len(series))

    df = pd.DataFrame(data, index=main_categories)
    ax = df.plot.bar(rot=25, stacked=True)
    plt.title(f'{sub_column_name}')

    plt.savefig(f'images/{sub_column_name.strip("? ")}.png', dpi=300)
    # plt.show()
    return



if __name__ == "__main__":
    # Load lengthy data from json file
    with open("supplementary_data.json", "r") as info:
        suppl_data = json.load(info)

    questions = suppl_data['questions']
    program_spellings = suppl_data['program_spellings']
    program_keywords = suppl_data['program_keywords']
    program_abbreviations = suppl_data['program_abbreviations']

    programs = list(program_spellings.keys())
    df = pd.read_excel('cleaned_dataset.xlsx')
    df[questions["master_program"]] = df[questions["master_program"]].map(program_abbreviations)

    few_answers = list(filter(lambda question: len(set(df[questions[question]])) <= 11, questions.keys()))
    numeric_columns = df.select_dtypes('float64')

    # code for bar plots of predefined columns

    # for key in few_answers:
    #     bar_plot(df, questions["master_program"], questions[key], program_abbreviations)

    # code for boxplots of predefined combinations of columns

    # for i, column_name in enumerate(numeric_columns):
    #     for j, category in enumerate(few_answers):
    #         plt.figure()
    #         ax = sns.boxplot(data=df, x=questions[category], y=column_name)
    #         ax.set_title(column_name)
    #         ax.set_ylabel('')
    #         title_string = column_name.strip(' ?')
    #         # ax.get_figure().savefig(f'images/{title_string}_groupedby_{category}.png', dpi=300)
    
    # code for scatterplots
    # numeric_columns = list(numeric_columns)
    # for i, numeric_1 in enumerate(numeric_columns):
    #     for numeric_2 in numeric_columns[i+1:]:
    #         ax = sns.scatterplot(data=df, x=numeric_1, y=numeric_2)
    #         # ax.set_title()
    #         plt.show()






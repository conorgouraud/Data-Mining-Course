import pandas as pd

def text_to_int(dataframe, column):
    '''
    Function to change object datatype of a specified column to int datatype.

    Arguments
    dataframe : pandas.dataframe object to change
    column : str; name of dataframe column to change.

    Returns
    todo
    '''
    new_df = pd.DataFrame()


if __name__ == '__main__':

    df = pd.read_csv('ODI-2023.csv', sep=';')
    # for row in df:
    #     print(row)
    print(df.dtypes)
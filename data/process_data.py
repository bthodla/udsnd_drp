# imports
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_file_path, categories_file_path):
    """
    A function that: 
    * accepts two CSV filenames as input
    * loads data from input filenames into Pandas dataframes
    * merges the two dataframes into one

    :param messages_file_path: (str) Messages CSV file
    :param categories_file_path: (str) Categories CSV file

    :return: a Pandas dataframe that merges the data from the two input files
    """

    messages = pd.read_csv(messages_file_path)
    categories = pd.read_csv(categories_file_path)
    
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    """
    A function that cleans the input dataframe for use by an ML model
    
    :param df: (pandas.core.frame.DataFrame) Merged dataframe returned by the 
                load_data() function

    :return: a Pandas dataframe that has been cleaned to be used by an ML model
    """

    # Create a dataframe of the individual category columns
    categories = df['categories'].str.split(";", expand = True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # Use this row to extract a list of new column names for categories.
    category_colnames = [r[:-2] for r in row]

    # Rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:

        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the original categories column from the input dataframe
    df.drop('categories', axis = 1, inplace = True)

    # Concatenate the input dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, db_file_name):
    """
    A function that saves the input dataframe to an SQLITE database

    :param df: (pandas.core.frame.DataFrame) Cleaned data returned by the 
                clean_data() function
    :param db_file_name: (str) File path of the SQLITE Database into which the 
                cleaned data is to be saved
    :return: None
    """
    
    engine = create_engine('sqlite:///{}'.format(db_file_name)) 
    # Extract the filename from db_file_name
    file_name = db_file_name.split("/")[-1]
    table_name = file_name.split(".")[0]
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, db_filepath = sys.argv[1:]

        print('Loading data...\n...Messages: {}\n...Categories: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n...Database: {}'.format(db_filepath))
        save_data(df, db_filepath)
        
        print('Cleaned input data and saved it to database')
    
    else:
        print('Usage:')
        print('\tpython process_data.py <Messages Data File Name> '\
            '<Categories Data File Name>  <DB File Name>')
        print('Example:')
        print('\tpython process_data.py disaster_messages.csv '\
            'disaster_categories.csv DisasterResponse.db')
        print('Please be mindful of the sequence of the parameters')


# run
if __name__ == '__main__':
    main()
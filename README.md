# Project GitHub Name: udsnd_drp
## Project Name: Udacity Data Science Nanodegree - Disaster Response Piplelines Project

In this project, I was provided with two types of data:
* Messages relating to disaster(s) that people were experiencing
* Categories of disasters

These two datasets were provided in CSV format.

The objectives of this project are as follows:
* Load the above data programmatically into Pandas dataframes
* Merge the two datasets into one
* Clean the data
* Load the cleaned data into an SQLITE database

* In another process, load the data from the SQLITE database into Pandas dataframes
* Split the data into training and test sets
* Build a text processing pipeline and a machine learning pipeline
* Train and tune the model using GridSearchCV
* Output the results of the test set
* Export the final model as a pickle file

* Finally, run the web
* app after making sufficient modifications to properly point to the SQLITE database and the saved model file

Code for this process is contained in <Project Root Dir>/data/process.py

Please run this code using the following command:

Usage:
        python process_data.py <Messages Data File Name> <Categories Data File Name>  <DB File Name>
Example:
        python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

Please be mindful of the sequence of the parameters

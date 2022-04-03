# Project GitHub Name: udsnd_drp
## Project Name: Udacity Data Science Nanodegree - Disaster Response Piplelines Project

### Summary
This project helps a disaster management organization analyze the messages that it receives to build a model that classifies disaster messages.


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


* Finally, make modifications to the web app to:
  * properly point to the SQLITE database and the saved model file
  * add additional visuals

Code for this process is contained in <Project Root Dir>/data/process.py

Please run this code using the following command:

### Loading and cleaning the data
Usage:

&nbsp;&nbsp;&nbsp;&nbsp;python process_data.py "Messages Data File Name" "Categories Data File Name" "DB File Name"

Example:

&nbsp;&nbsp;&nbsp;&nbsp;python ..\data\process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

Please be mindful of the sequence of the parameters

### Training the model
Usage:

&nbsp;&nbsp;&nbsp;&nbsp;python ..\models\train_classifier.py "DB File Name" "Model File Name"

Example:
&nbsp;&nbsp;&nbsp;&nbsp;python train_classifier.py ..\Data\DisasterResponse.db classifier.pkl

Please be mindful of the sequence of the parameters

### Running the web application
Usage:

&nbsp;&nbsp;&nbsp;&nbsp;python ..\app\run.py

### Project Code Structure
Project Root
* app
  * templates
    * go.html: this is page that shows classification results
    * master.html: this is the main page of the web app
  * run.py: this contains the Flask file that runs the app
* data
  * disaster_categories.csv: contains disaster category data
  * disaster_messages.csv: contains disaster messages data
  * DisasterResponse.db: the SQLITE database into which the cleaned and merged data is stored
  * process_data.py: code to load, clean, merge and save the data from the above CSV files
* models
  * classifier.pkl: file into which the trained model is saved
  * train_classifier.py: code to train and save the model
* notebooks
  * contains two notebooks to test code before they are written as Python programs
* README.md: this file
* LICENSE: Open Source License terms
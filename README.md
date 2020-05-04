# Mortgage_Default
 This project analyzes the characteristics of delinquent loans and borrowers at the time of origination. The period covered in this study is between 2013 and 2018.
 
 1. Folder 'Code' containing all codes used in this project.   
   a. Preprocessing.ipynb imports, cleans the data, and merges different dataframes  
   b. Modeling.ipynb applies several machine learning methods to classify delinquent vs performing loans  
 
 These two files run a subsample of the original data to demonstrate that the codes have no error.  
 
 2. Since the original data is too big to run on local machine. Pycharm was used to connect to an AWS EC2 instance to run the data on the whole population. Folder 'Archive - Pycharm format' includes the similar codes as the sample codes with different input data and naming schemes for output data.  
 
 3. Folder 'Model Scores' includes all scores resulting from running the 2.Modeling.py on the whole population.  

### Real world user classification project for a Ukrainian telecommunications company aimed  at predicting future customer behavior. 

In this program I build a deep learning model with 95% classification accuracy. However, most of the code is devoted to data wrangling (data cleaning, preparation, transformation). 


There are four csv data files:

tabular_data.csv,
hashed_data.csv,
train_target.csv,
test_target.csv.

The objective is to build a model that will indicate whether a customer is a parent.

This is a binary classification problem:

"1" means the customer has children younger than 14 (i.e. he/she is a parent)
"0" means otherwise (the customer is not a parent)



The files tabular_data.csv and hashed_data.csv contain descriptors for 4871 customers ('ID' pertains to a customer id).

train_target.csv is the training data set.

test_target.csv is the test data used to test the model on the previously unseen data.


#### More in detail:


The file 'tabular_data.csv' contains numerical information on the activity of the customer over three time periods.

'Period' is the period number (the periods are consecutive, 1 being the earliest);
'ID' is customer id;
'V1' – 'V43' is the activity data of the customer over a given period.

The file 'hashed_data.csv' contains hashed values of one categorical variable for a customer.

'ID' is customer id;
'HASH' is the hash value of a categorical variable.

The file 'train_target.csv' contains target (label) data.

'ID' is customer id;
'TARGET' is the target value (1 belongs to the parent class, 0 is not in the parent class).

The file 'test_target.csv' contains a list of customers the model should make predictions for.

'ID' is customer id;
SCORE is the probability of the customer belonging to the parent class (class 1). 

The performance of the model should be evaluated using the AUROC metric.

One should use the data from the files tabular_data and hashed_data. Having built a respective model, one should evaluate the SCORE from the 'test_target' file, that is the probability that the customer belongs to the parent class (in general, irrespective of the time period). 



#### Requirements

Python == 3.5

Tensorflow == 2.0

keras == 2.3.1
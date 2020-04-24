#import libraries to work with arrays and dataframes
import pandas as pd
import numpy as np



#create dataframe objects from csv files
td = pd.read_csv('tabular_data.csv')
hd = pd.read_csv('hashed_data.csv')
train_t = pd.read_csv('train_target.csv')
test_t = pd.read_csv('test_target.csv')

#convert categorical variables to numerical
hd_cat = pd.get_dummies(hd)

#present the data in a more convenient form (no many-to-many relationships)
hd_cat_comb = hd_cat.groupby(['ID']).sum()

#join tabular and hashed data into one table 
df_full = td.merge(hd_cat_comb,how='outer',on='ID')



#fill the Null values with the column mean values
df_nonull = df_full.fillna(df_full.mean())

#import a tool to rescale data and instantiate it
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

#define the input(feature) data
X = pd.DataFrame(mms.fit_transform(df_nonull),columns=df_nonull.columns).drop(['ID'],axis=1)

#for every user assign the same value of the Target for each Period (either 0 or 1)
y_train =df_nonull.merge(train_t,on='ID')['TARGET']

#define test targets
y_test =df_nonull.merge(test_t,on='ID')[['ID','SCORE']]

#split the data into train and test parts
X_train = X[:11583]
X_test = X[11583:]



#import deep learning classes (employing Keras API)
from keras import backend as K
from keras.layers import Dense,BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
X_train_train,X_train_test,y_train_train,y_train_test = train_test_split(X_train,y_train,test_size=0.28)

#build a fully-connected neural network with batch normalization layers
K.clear_session()

model = Sequential()

model.add(Dense(10,input_shape = X_train_train.shape[1:],activation='relu',kernel_initializer='normal'))

model.add(BatchNormalization())

model.add(Dense(5,activation='relu',kernel_initializer='normal'))

model.add(BatchNormalization())

model.add(Dense(1,activation = 'sigmoid'))

model.compile(metrics=['accuracy'],optimizer = 'rmsprop',loss = 'binary_crossentropy')



#train the model
model.fit(X_train_train,y_train_train,batch_size=64,epochs = 300,validation_data = (X_train_test,y_train_test),callbacks=[EarlyStopping('loss',patience = 5)])

#check the performance of the model

from sklearn.metrics import roc_auc_score
y_train_pred = model.predict(X_train_test)
roc_auc_score(y_true = y_train_test,y_score=y_train_pred)

y_score = model.predict(X_test)

interm = pd.DataFrame(X_test['PERIOD'],columns = ['PERIOD'])
interm['SCORE']=y_score

check = interm.join(df_nonull['ID'])

check1 = check.groupby('ID').mean()



y_test_df=pd.DataFrame(y_test['ID'])

test_t['SCORE']=check1['SCORE'].values

check1['SCORE'].values.shape

#final result
test_t

test_t.to_csv(r'KravtsivIvan_test.txt',index=False)




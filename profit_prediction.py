# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#-*- coding=utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
#import seaborn as sns

#import attributes table
attrs =  pd.read_excel('attrs.xlsx',sheetname='Sheet1', encoding = 'utf-8') 
sku_attrs = attrs[['sku_id','attr_name','attr_value']]

#transform original table to pivot_table 
a = pd.pivot_table(sku_attrs[['sku_id','attr_name','attr_value']],
                   index=['sku_id'],columns=['attr_name'],
                    values=['attr_value'],fill_value='NaN',aggfunc='max')
a.columns = a.columns.droplevel(level=0)
a = a.reset_index(drop=False)


#replace wrong information with correct data while transforming
a.replace(42928,'7-12',inplace=True)
a.replace(u'其他',u'其它',inplace=True)



#import profit table
sku_profit = pd.read_excel('sku_profit.xlsx',sheetname = 'Sheet123')
sku_profit['sku_id'] = sku_profit['item_sku_id']
sku_profit.drop(['dt','item_third_cate_name','item_sku_id','gmv','cost_tax',
'income','grossfit','gross_sales','rebate_amunt_notax',
'adv_amount','store_fee','deliver_fee'], axis = 1, inplace = True)

#put the last column to the very front
cols = sku_profit.columns.tolist()
cols = cols[-1:]+cols[:-1]
sku_profit = sku_profit[cols]

#extract the mean sku_id profit table
average_profit = sku_profit.groupby('sku_id').mean()
average_profit.reset_index(inplace=True)


#merge attributes table and mean profit table based on sku_id
final_npp = pd.merge(a,average_profit, how = 'inner', on = 'sku_id')
final_npp.drop('sku_id', axis = 1, inplace=True)


#label encoder method to handle the categorical columns except the net_profit column
for attribute in final_npp.columns.difference(['net_profit']):
    le = preprocessing.LabelEncoder()
    final_npp[attribute] = le.fit_transform(final_npp[attribute])
    
#sns.distplot(final_npp['net_profit'] )


#print(final_npp.loc[final_npp['net_profit'] < -1500000])
#final_npp.to_csv('final_npp.csv', encoding = 'utf-8')
final_npp.drop(final_npp.index[[39,52]],inplace = True)
#print(final_npp.loc[final_npp['net_profit'] < -1700000])




#train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_npp.drop('net_profit',axis=1), 
                                                    final_npp['net_profit'], 
                                                    test_size=0.30, 
                                                    random_state = 101)
#X_train = X_train.as_matrix()
#y_train = y_train.as_matrix()
#X_test = X_test.as_matrix()
#y_test = y_test.as_matrix()



'''
#build linear regression model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
'''



#build random forest model
from sklearn.ensemble import RandomForestRegressor
  
rfr = RandomForestRegressor(n_estimators = 100, 
                              max_features = 13,
                              max_depth=4,
                              #min_samples_split=4,
                              oob_score=True,
                              n_jobs=-1)
rfr.fit(X_train, y_train)
predictions = rfr.predict(X_test)


#model evaluation
plt.scatter(y_test,predictions)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))































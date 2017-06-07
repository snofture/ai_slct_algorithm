# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 15:42:10 2017

@author: limen
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
#import seaborn as sns
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

from sklearn.utils import check_array
def mean_absolute_percentage_error(y,p):
    y = check_array(y)
    p = check_array(p)
    return np.mean(np.abs((y-p)/y))*100

def edit_package_column(aarray):
    for i in aarray:
        if len(i) > 4:
            aarray.remove(i)
    return aarray

if __name__ == '__main__':
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
    a.replace(42741,'1-6',inplace=True)
    a.replace(42928,'7-12',inplace=True)
    a.replace(u'其他',u'其它',inplace=True)
    #a.drop(u'适用场景',axis = 1, inplace=True)
    #a.drop(u'功能饮料', axis = 1, inplace = True)
    #a.drop(u'是否含糖', axis = 1, inplace = True)
    #a.drop(u'茶饮料系列', axis = 1, inplace = True)
    a.drop([u'适用场景',u'功能饮料',u'是否含糖',u'茶饮料系列'],axis = 1, inplace = True)
    #c = a[u'包装'].map(lambda x: len(x))
    a[u'包装'] = a[u'包装'].edit_package_column()
    
    
    #import profit table
    sku_profit = pd.read_excel('sku_profit.xlsx',sheetname = 'Sheet123')
    sku_profit['sku_id'] = sku_profit['item_sku_id']
    sku_profit.drop(['dt','item_third_cate_name','item_sku_id','cost_tax',
    'income','grossfit','gross_sales','rebate_amunt_notax',
    'adv_amount','store_fee','deliver_fee'], axis = 1, inplace = True)
    
    #put the last column to the very front
    cols = sku_profit.columns.tolist()
    cols = cols[-1:]+cols[:-1]
    sku_profit = sku_profit[cols]
    
    #make the profit_rate column
    sku_profit = sku_profit[sku_profit['gmv'] != 0]
    sku_profit = sku_profit[sku_profit['net_profit'] < sku_profit['gmv']]
    sku_profit['profit_rate'] = sku_profit['net_profit']/sku_profit['gmv']*100
    #sku_profit[~np.isfinite(sku_profit)] = np.nan
    sku_profit.drop(['net_profit','gmv'], axis =1, inplace = True)
    
    #extract the mean sku_id profit table
    average_profit = sku_profit.groupby('sku_id').mean()
    average_profit.reset_index(inplace=True)
    
    
    #merge attributes table and mean profit table based on sku_id
    final_npp = pd.merge(a,average_profit, how = 'inner', on = 'sku_id')
    final_npp.drop('sku_id', axis = 1, inplace=True)
    #pd.options.display.float_format='{:,.0f}'.format
    final_npp['profit_rate'] = final_npp['profit_rate'].astype(int)
    final_npp.drop(final_npp.index[[38,412,530,85,380,395,483,535,497,509,3,449,450,533]],inplace = True)
    
    
    #label encoder method to handle the categorical columns except the net_profit column
    for attribute in final_npp.columns.difference(['profit_rate']):
        le = preprocessing.LabelEncoder()
        final_npp[attribute] = le.fit_transform(final_npp[attribute])
    
    
    #sns.distplot(final_npp['net_profit'] ) 观察net_profit分布情况
    
    
    #print(final_npp.loc[final_npp['net_profit'] < -1500]) label or conditional based selection
    #final_npp.to_csv('final_npp.csv', encoding = 'utf-8') to show final_npp content
    final_npp.drop(final_npp.index[[38,412,530]],inplace = True) #drop rows which have large noise based on index selection
    
    #print(final_npp.loc[final_npp['profit_rate'] < -1500])
    
    
    
    
    #train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(final_npp.drop('profit_rate',axis=1), 
                                                        final_npp['profit_rate'], 
                                                        test_size=0.30, 
                                                        random_state = 101)
    X_train = X_train.as_matrix()
    y_train = y_train.as_matrix()
    X_test = X_test.as_matrix()
    y_test = y_test.as_matrix()
    
    y_test[y_test == 0] = 1
    
    
    '''
    #build linear regression model
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)
    
    '''
    
    
    #build random forest model
    from sklearn.ensemble import RandomForestRegressor
      
    rfr = RandomForestRegressor(n_estimators = 55, 
                                  max_features = 8,
                                  max_depth=3,
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
    
    #y = check_array(y_test)
    #a = mean_absolute_percentage_error(y_test,predictions)
    print('MAPE:', mean_absolute_percentage_error(y_test,predictions))
    

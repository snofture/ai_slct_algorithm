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
import re
#import seaborn as sns
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

from sklearn.utils import check_array
def mean_absolute_percentage_error(y,p):
    y = check_array(y)
    p = check_array(p)
    return np.mean(np.abs((y-p)/y))*100

#import attributes table
attrs =  pd.read_excel('attrs.xlsx',sheetname='Sheet1', encoding = 'utf-8') 
sku_attrs = attrs[['sku_id','attr_name','attr_value']]

#transform original table to pivot_table 
a = pd.pivot_table(sku_attrs[['sku_id','attr_name','attr_value']],
                   index=['sku_id'],columns=['attr_name'],
                    values=['attr_value'],fill_value='None',aggfunc='max')
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
           
a[u'产品产地'] = a[u'产品产地'].apply(lambda x: re.sub('.*#\$','',x))
a[u'包装'] = a[u'包装'].apply(lambda x: re.sub('.*#\$','',x))
a[u'分类'] = a[u'分类'].apply(lambda x: re.sub('.*#\$','',x))
a[u'碳酸饮料分类'] = a[u'碳酸饮料分类'].apply(lambda x: re.sub('.*#\$','',x))
a[u'口味'] = a[u'口味'].apply(lambda x: re.sub('.*#\$','',x))

def juice_percentage(x):
    if (x[u'分类'] == u'果蔬汁') & (x[u'果汁成分含量'] == 'None'):
        return 1 
    elif (x[u'分类'] == u'果汁') & (x[u'果汁成分含量'] == 'None'):
        return u'100%以下'
    elif (x[u'分类'] == u'果汁') & (x[u'果汁成分含量'] == u'其它'):
        return u'100%以下'
    elif (x[u'分类'] == u'果味饮料') & (x[u'果汁成分含量'] == 'None'):
        return u'100%以下'
    elif (x[u'分类'] == u'果味饮料') & (x[u'果汁成分含量'] == u'其它'):
        return u'100%以下'
    else:
        return x[u'果汁成分含量']
    
a[u'果汁成分含量'] = a.apply(lambda x: juice_percentage(x), axis = 1)

#a[u'果汁成分含量'] = a.apply(lambda x: 1 if (x[u'分类'] == u'果蔬汁') & (x[u'果汁成分含量'] == Null) else x[u'果汁成分含量'] )

'''
def edit_package_column(i):
    #if len(i) > 4:
        i = i.split('#')[0]
        return i
'''

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
#sku_profit = sku_profit[sku_profit['gmv'] <= -1] #the number of gmv == 0 rows should be less than 35216
sku_profit_1 = sku_profit[sku_profit['gmv'] >= 1 ]
sku_profit_2 = sku_profit[sku_profit['gmv'] <= -1 ]
sku_profit = pd.concat([sku_profit_1,sku_profit_2])
sku_profit = sku_profit[sku_profit['net_profit'] < sku_profit['gmv']]
#sns.distplot(final_npp['net_profit'] ) 观察net_profit分布情况
sku_profit['profit_rate'] = sku_profit['net_profit']/sku_profit['gmv']*100
#sku_profit[~np.isfinite(sku_profit)] = np.nan
sku_profit.drop(['net_profit','gmv'], axis =1, inplace = True)

#drop off the ones with sku_profit_rate<-500 and >500
sku_profit = sku_profit[sku_profit['profit_rate'] > -300]
sku_profit = sku_profit[sku_profit['profit_rate'] < 300]


#extract the mean sku_id profit table
average_profit = sku_profit.groupby('sku_id').mean()
average_profit.reset_index(inplace=True)


#merge attributes table and mean profit table based on sku_id
final_npp = pd.merge(a,average_profit, how = 'inner', on = 'sku_id')
#final_npp.drop('sku_id', axis = 1, inplace=True)
final_npp['profit_rate'] = final_npp['profit_rate'].astype(int)
#final_npp.drop(final_npp.index[[412,530,85,380,395,483,535,497]],inplace = True)
final_npp = final_npp[final_npp['profit_rate'] > -150]

#final_npp.to_csv('final_npp.csv', encoding = 'utf-8') to show final_npp content
#f = final_npp.iloc[38]
final_npp.drop(final_npp.index[[38,412,530]],inplace = True) #drop rows which have large noise based on index selection



#import sku_price table
sku_price = pd.read_csv('drinks.csv')
#print(sku_price.columns)
sku_price.drop(u'Unnamed: 0', axis = 1, inplace = True)
sku_price['sku_id'] = sku_price['item_sku_id']
sku_price.drop(['item_first_cate_cd','item_second_cate_cd',
                'item_third_cate_cd','item_sku_id'], axis = 1, inplace = True)


net_profit_percent = pd.merge(final_npp,sku_price, how = 'inner', on = 'sku_id')
net_profit_percent.drop('sku_id', axis = 1, inplace=True)
cols = net_profit_percent.columns.tolist()
cols = cols[-1:]+cols[:-1]
net_profit_percent = net_profit_percent[cols]



net_profit_percent = pd.get_dummies(net_profit_percent, columns=[u'产品产地',
                                                                 u'分类', 
                                                                 u'包装',
                                                                 u'包装件数',
                                                                 u'单件容量',
                                                                 u'口味',
                                                                 u'果汁成分含量',
                                                                 u'碳酸饮料分类',
                                                                 u'进口/国产'],
    prefix=['origin', 'classification','package','pack unit','unit vol','taste'
            ,'juice percent','carbohydra','import/export'])
'''
#from sklearn.preprocessing import LabelEncoder
#label encoder method to handle the categorical columns except the net_profit column
for attribute in net_profit_percent.columns.difference(['profit_rate','price']):
    le = preprocessing.LabelEncoder()
    net_profit_percent[attribute] = le.fit_transform(net_profit_percent[attribute])
 '''
#one-hot encoding method to hand categorical/discrete features
#for attribute in net_profit_percent.columns.difference(['profit_rate','price']): 
#enc = preprocessing.OneHotEncoder(categorical_features='all',
       #handle_unknown='error', n_values='auto', sparse=True)
#c = enc.fit_transform(net_profit_percent[[u'产品产地',u'分类']])
#cols_to_transform = [ u'产品产地', u'类别' ]
#df_with_dummies = pd.get_dummies(  cols_to_transform )


'''
#one-hot encoding
from sklearn.feature_extraction import DictVectorizer
 
def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict()).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df
'''    
    
    
if __name__ == '__main__':    
    
    #net_profit_percent = encode_onehot(net_profit_percent, [u'产品产地'])
    
    
    #train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(net_profit_percent.drop('profit_rate',axis=1), 
                                                        net_profit_percent['profit_rate'], 
                                                        test_size=0.30, 
                                                        random_state = 101)
    X_train = X_train.as_matrix()
    y_train = y_train.as_matrix()
    X_test = X_test.as_matrix()
    y_test = y_test.as_matrix()
    
    y_test[y_test == 0] = 1
    
    
    
    
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
    '''
    
    #model evaluation
    plt.scatter(y_test,predictions)
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    print('MAPE:', mean_absolute_percentage_error(y_test,predictions))
    
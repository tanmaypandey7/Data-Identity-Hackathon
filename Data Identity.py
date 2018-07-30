import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
start=time.time()
test=pd.read_csv(r'E:\Data Science\Competitons Data\AV data identity\test_2nAIblo.csv')
train=pd.read_csv(r'E:\Data Science\Competitons Data\AV data identity\train_HK6lq50.csv')
sample=pd.read_csv(r'E:\Data Science\Competitons Data\AV data identity\sample_submission_vaSxamm.csv')
test1=test
train_pred=train['is_pass']
train=train.drop(['id','test_id','total_programs_enrolled','trainee_id','is_pass'],axis=1)
train['difficulty_level']=train['difficulty_level'].replace(['easy','intermediate','hard','vary hard'],['0','1','2','3']).astype(int)
train['education']=train['education'].replace(['No Qualification', 'High School Diploma','Matriculation', 'Bachelors', 'Masters'],['0','1','2','3','4']).astype(int)
train['test_type']=train['test_type'].replace(['offline', 'online'],['0','1']).astype(int)

train['gender']=train['gender'].replace(['M', 'F'],['0','1']).astype(int)

train['is_handicapped']=train['is_handicapped'].replace(['N', 'Y'],['0','1']).astype(int)
train['program_id']=train['program_id'].replace(['S_1', 'S_2', 'T_1', 'T_2', 'T_3', 'T_4', 'U_1', 'U_2', 'V_1', 'V_2', 'V_3', 'V_4', 'X_1', 'X_2', 'X_3', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Z_1', 'Z_2', 'Z_3'], [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']).astype(int)
train['program_type']=train['program_type'].replace(['S', 'T', 'U', 'V', 'X', 'Y', 'Z'],['1','2','3','4','5','6','7']).astype(int)

train.loc[train['program_duration'] < 117, 'program_duration'] = 0
train.loc[(train['program_duration'] >= 117) & (train['program_duration'] <= 121), 'program_duration'] = 1
train.loc[(train['program_duration'] > 121) & (train['program_duration'] <=131), 'program_duration'] = 2
train.loc[(train['program_duration'] > 131) & (train['program_duration'] <= 134), 'program_duration'] = 3
train.loc[(train['program_duration'] > 134), 'program_duration'] = 4
train['program_duration']=train['program_duration'].astype(int)
train['age']=train['age'].fillna(np.mean(train['age']))
train.loc[train['age'] < 17, 'age'] = 0
train.loc[(train['age'] >= 17) & (train['age'] <= 28), 'age'] = 1
train.loc[(train['age'] > 28) & (train['age'] <=39), 'age'] = 2
train.loc[(train['age'] > 39) & (train['age'] <= 45), 'age'] = 3
train.loc[(train['age'] > 45), 'age'] = 4
train['age']=train['age'].astype(int)
train['trainee_engagement_rating']=train['trainee_engagement_rating'].fillna('1').astype(int)
train['points']=train['program_type']+train['program_id']
# train.corr().plot(kind='heatmap',subplots='true',layout=(4,4))
# plt.show()
# k=train.drop(['is_pass'],axis=1)
# k=list(k.columns)
# for i in k:
#     print(train[[i, 'is_pass']].groupby([i], as_index=False).mean().sort_values(by='is_pass',ascending=False))
# train['program_id', 'program_type', 'test_type', 'difficulty_level', 'gender', 'education','trainee_engagement_rating']=train['program_id', 'program_type', 'test_type', 'difficulty_level', 'gender', 'education','trainee_engagement_rating']
# train['trainee_engagement_rating'].plot(kind='hist')
# plt.show()
# train['trainee_engagement_rating'].plot(kind='hist',cumulative=True)
# plt.show()

test=test.drop(['id','test_id','total_programs_enrolled','trainee_id'],axis=1)
test['difficulty_level']=test['difficulty_level'].replace(['easy','intermediate','hard','vary hard'],['0','1','2','3'])
test['education']=test['education'].replace(['No Qualification', 'High School Diploma','Matriculation', 'Bachelors', 'Masters',],['0','1','2','3','4'])
test['test_type']=test['test_type'].replace(['offline', 'online'],['0','1'])

test['gender']=test['gender'].replace(['M', 'F'],['0','1'])

test['is_handicapped']=test['is_handicapped'].replace(['N', 'Y'],['0','1'])
test['program_id']=test['program_id'].replace(['S_1', 'S_2', 'T_1', 'T_2', 'T_3', 'T_4', 'U_1', 'U_2', 'V_1', 'V_2', 'V_3', 'V_4', 'X_1', 'X_2', 'X_3', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Z_1', 'Z_2', 'Z_3'], [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22'])

test.loc[test['program_duration'] < 117, 'program_duration'] = 0
test.loc[(test['program_duration'] >= 117) & (test['program_duration'] <= 121), 'program_duration'] = 1
test.loc[(test['program_duration'] > 121) & (test['program_duration'] <=131), 'program_duration'] = 2
test.loc[(test['program_duration'] > 131) & (test['program_duration'] <= 134), 'program_duration'] = 3
test.loc[(test['program_duration'] > 134), 'program_duration'] = 4

test['program_type']=test['program_type'].replace(['S', 'T', 'U', 'V', 'X', 'Y', 'Z'],['1','2','3','4','5','6','7'])

test['age']=test['age'].fillna(np.mean(test['age']))
test.loc[test['age'] < 17, 'age'] = 0
test.loc[(test['age'] >= 17) & (test['age'] <= 28), 'age'] = 1
test.loc[(test['age'] > 28) & (test['age'] <=39), 'age'] = 2
test.loc[(test['age'] > 39) & (test['age'] <= 45), 'age'] = 3
test.loc[(test['age'] > 45), 'age'] = 4
test['points']=test['program_type']+test['program_id']
test['trainee_engagement_rating']=test['trainee_engagement_rating'].fillna('1')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
clf=DecisionTreeClassifier()
clf.fit(train,train_pred)


pred=clf.predict(test)
score=accuracy_score(pred,sample.iloc[:,1])
print(score)


#PREPARING FOR EXPORT TO EXCEL

pred_df=pd.DataFrame(pred,columns=['is_pass'])
pred_df['id']=test1['id']
columnsTitles=["id","is_pass"]
pred_df=pred_df.reindex(columns=columnsTitles)


# from pandas import ExcelWriter
# writer= ExcelWriter('DataIdentity.xlsx',engine='xlsxwriter')
# pred_df.to_excel(writer,sheet_name='sheet 1',index=False)
# writer.save()

pred_df.to_csv('DataIdentity2.csv',index=False)
stop=time.time()
print(stop-start)

# #
import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv('paysim_data.csv')
df['hour'] = df['step'] % 24
df['amount_log'] = np.log1p(df['amount'])
df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

type_mapping = {'CASH_OUT':0,'PAYMENT':1,'CASH_IN':2,'TRANSFER':3,'DEBIT':4}

cash_out = df[df['type']=='CASH_OUT']
fraud_co = cash_out[cash_out['isFraud']==1]
legit_co = cash_out[cash_out['isFraud']==0]
print('CASH_OUT fraud count', len(fraud_co), 'legit count', len(legit_co))
print('fraud balance_diff_dest unique sample:', fraud_co['balance_diff_dest'].describe())
print('legit balance_diff_dest unique sample:', legit_co['balance_diff_dest'].describe())
print('fraud balance_diff_orig describe:', fraud_co['balance_diff_orig'].describe())
print('avg amount ratio fraud', fraud_co['amount_to_balance_ratio'].mean())
print('avg amount ratio legit', legit_co['amount_to_balance_ratio'].mean())
print('type mapping sample', type_mapping['CASH_OUT'])

# find nearest dataset entries to sample pattern
target = {'amount':200000, 'hour':2, 'balance_diff_orig':-200000, 'balance_diff_dest':0, 'amount_to_balance_ratio':200000/(210000+1), 'oldbalanceOrg':210000, 'newbalanceOrig':10000}

def dist(row):
    return abs(row['amount']-target['amount']) + abs(row['hour']-target['hour'])*1000 + abs(row['balance_diff_orig']-target['balance_diff_orig'])/1000 + abs(row['balance_diff_dest']-target['balance_diff_dest'])*1000 + abs(row['amount_to_balance_ratio']-target['amount_to_balance_ratio'])*1000

cand = cash_out.copy()
cand['dist'] = cand.apply(dist, axis=1)
print(cand.sort_values('dist').head(10)[['isFraud','amount','hour','balance_diff_orig','balance_diff_dest','amount_to_balance_ratio','oldbalanceOrg','newbalanceOrig']])

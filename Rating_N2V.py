import gzip
from collections import defaultdict
from random import randint                                                                                                                                                                                                                    
from math import sqrt
from math import fabs
import operator
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

# 39882 128 
# N2v MODEL

u_b_dict = defaultdict()
user_buisness = defaultdict(set)
businessCount = defaultdict(int)
totalPurchases = 0
user_d = defaultdict()
business_d = defaultdict()

train_data = defaultdict()

counter = 0
for l in readGz("train.json.gz"):
  a,b,c = l['userID'],l['businessID'],l['rating']
  u_b_dict[a] = 1
  u_b_dict[b] = 1
  user_d[a] =1
  business_d[b] = 1
  train_data[(a,b)] = float(c)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
  user_buisness[a].add(b)

user = []
for k,v in user_d.iteritems():
  user.append(k)
business = []
for k,v in business_d.iteritems():
  business.append(k)

u_b_index = []
for k,v in u_b_dict.iteritems():
  u_b_index.append(k)

for i in range(len(u_b_index)):
  u_b_dict[u_b_index[i]] = i

deep1t = np.array(pd.read_csv('Embeddings_8_35.txt', delimiter="\s+",header=None))
matrixSize = max(deep1t[:,0]).astype(int)+1
deep1 = np.zeros([matrixSize,128])

for i in range(0, deep1t.shape[0]):
    deep1[deep1t[i,0].astype(int)] = deep1t[i,1:129];

print '1st embedding loaded'

X=[]
Y=[]

for k,v in train_data.iteritems() :
  a,b  = k
  X.append(np.multiply(deep1[u_b_dict[a],:],deep1[u_b_dict[b],:]))
  Y.append(v)

print 'data modelled'

X = np.array(X)
# X = np.reshape(X,(X.shape[0],1))
Y = np.array(Y)
print X.shape, Y.shape


model =LinearRegression()
model.fit(X,Y)

# ir = IsotonicRegression()
# ir.fit(X, Y)
predicted = model.predict(X)
print mean_squared_error(Y,predicted)

clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X, Y)

predicted = clf.predict(X)
print mean_squared_error(Y,predicted)

new_u = 0
new_b = 0
predictions = open("predictions_Rating_N2V_SVR.txt", 'w')
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  if  u not in u_b_dict:
    new_u +=1
    predictions.write(u + '-' + i + ",4.184485\n")  
  else:
    if i not in u_b_dict:
      new_b +=1
      predictions.write(u + '-' + i + ",4.184485\n")
    else:
      a = np.multiply(deep1[u_b_dict[u],:],deep1[u_b_dict[i],:])
      a = np.array(a)
      a = np.reshape(a,(1,-1))
      pred = clf.predict(a)
      predictions.write(u + '-' + i + ",{}\n".format(str(pred[0]))) 
predictions.close()

print new_u,new_b

# ### END of Node2Vec Model

# allRatings = []
# userRatings = defaultdict(list)
# for l in readGz("train.json.gz"):
#   user,business = l['userID'],l['businessID']
#   allRatings.append(l['rating'])
#   userRatings[user].append(l['rating'])


# print allRatings[:1000]

# globalAverage = sum(allRatings[:len(allRatings)]) / (len(allRatings))

# print globalAverage

# userAverage = {}
# for u in userRatings:
#   userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

# test = allRatings[len(allRatings)/2:]
# RMSE = 0.0

# # Part 6 - New Model
# user_dict = defaultdict(set)
# business_dict = defaultdict(set)
# user_buisness_rating = defaultdict(dict)
# user =[]
# business=[]
# for l in readGz("train.json.gz"):
#   a,b = l['userID'],l['businessID']
#   user.append(a)
#   business.append(b)
#   user_buisness_rating[a][b] = l['rating']

# for i in range(len(user)):
#   a = user[i]
#   b = business[i]
#   user_dict[a].add(b)
#   business_dict[b].add(a)
  
# bu = defaultdict(float)
# bi = defaultdict(float)
# a = 0.0
# lam = 1.0

# def loss(R):
#   global a
#   global bu
#   global bi
#   loss =0.0
#   for i in range(len(user)/2,len(user)):
#     x = user[i]
#     y = business[i]
#     loss += (a + bu[x] + bi[y] - R[x][y])**2

#   loss /= 1.0*(len(user)-len(user)/2)
#   loss = sqrt(loss)
#   return loss

# def training_loss(R):
#   global a
#   global bu
#   global bi
#   loss =0.0
#   for i in range(len(user)):
#     x = user[i]
#     y = business[i]
#     loss += (a + bu[x] + bi[y] - R[x][y])**2

#   loss /= 1.0*(len(user))
#   loss = sqrt(loss)
#   return loss


# def update(R,lam):
#   global a
#   global bu
#   global bi
#   RMSE = training_loss(R)
#   prev_RMSE = 5.0
#   counter = 0
#   while ( fabs(prev_RMSE - RMSE) > 0.0001 ):
#     temp = 0.0
#     for i in range(len(user)):
#       x = user[i]
#       y = business[i]
#       temp += 1.0*(R[x][y] - (bu[x] +bi[y]))
#     temp /= (1.0*len(user))
#     a = temp

#     for u,l in user_dict.iteritems():
#       temp = 0.0
#       for i in l:
#         temp += 1.0*(R[u][i] - (a +bi[i]))
      
#       temp /= 1.0*(lam+len(l))
#       bu[u] = temp

#     for b,l in business_dict.iteritems():
#       temp = 0.0
#       for i in l:
#         temp += 1.0*(R[i][b] - (a +bu[i]))
#       temp /= 1.0*(lam+len(l))
#       bi[b] = temp
#     prev_RMSE =RMSE
#     RMSE = training_loss(R)
#     counter+=1
#     # print 'RMSE at convergence step {} Training : {}, Validation : {}'.format(counter,RMSE,loss(R))
#     # print 'Aplha = {} '.format(a)

#   print '\nRMSE at convergence. Lambda = {} Training : {}, Validation : {}\n'.format(lam,RMSE,loss(R))


# update(user_buisness_rating,lam)

# sorted_BU = sorted(bu.iteritems(), key=operator.itemgetter(1))
# sorted_BI = sorted(bi.iteritems(), key=operator.itemgetter(1))

# print 'Alpha = {}\nTen highest B_u users = {}\nTen highest B_i business = {}\n'.format(a,sorted_BU[-10:],sorted_BI[-10:])

# print 'User with smallest B_u is {}, and largest B_u is {}'.format(sorted_BU[0],sorted_BU[-1])
# print 'User with smallest B_i is {}, and largest B_i is {}'.format(sorted_BI[0],sorted_BI[-1])

# lam = [0.01,0.1,1,10,100]
# for l in lam:
#   update(user_buisness_rating,l)

# update(user_buisness_rating,10 )

# predictions = open("predictions_Rating_all_l10.txt", 'w')
# for l in open("pairs_Rating.txt"):
#   if l.startswith("userID"):
#     #header
#     predictions.write(l)
#     continue
#   u,i = l.strip().split('-')
#   # if u not in user_dict or b not in business_dict:
#   #   predictions.write(u + '-' + i + ',' + str(globalAverage) + '\n')  
#   # else:
#   prediction = a + bu[u] + bi[i]
#   if prediction > 5.0:
#     predictions.write(u + '-' + i + ',' + str(5.0) + '\n')
#   else:
#     if prediction < 0.0:
#       predictions.write(u + '-' + i + ',' + str(0.0) + '\n')
#     else:
#       predictions.write(u + '-' + i + ',' + str(prediction) + '\n')
  
# predictions.close()

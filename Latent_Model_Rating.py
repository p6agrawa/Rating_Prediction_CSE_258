
'''
Implements Alternating Least Squares (ALS) to create a recommender system for a subset of the Netflix dataset.
'''
import gzip
from collections import defaultdict
from random import randint                                                                                                                                                                                                                    
from math import sqrt
from math import fabs
import operator
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from scipy.sparse import csc_matrix

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)


def cf_ridge_regression(target_matrix, fixed_matrix, data, _lambda, verbose=False):
    '''
    Solves a ridge regression problem using a closed form solution:
        w_i = (X'X + lambda * I)^-1 X'y
    
    for all i in the target matrix.
    '''
    for j in xrange(target_matrix.shape[1]):
        if verbose and (j < 10 or j % 100 == 0):
            print '\tTarget column: {0}'.format(j)

        # Get only the non-missing entries for this movie or user
        nonzeros = data[:,j].nonzero()[0]
        y = data[nonzeros, j].todense()
        X = fixed_matrix[:, nonzeros].T

        #print 'nonzero: {0}'.format(nonzeros)
        #print 'y: {0} X: {1}'.format(y.shape, X.shape)
        #print 'inv: {0}'.format(np.linalg.inv(X.T.dot(X) + _lambda * np.eye(X.shape[1])).shape)
        #print "X'y: {0}".format(X.T.dot(y).shape)
        #print 'full: {0}'.format(np.squeeze(np.linalg.inv(X.T.dot(X) + _lambda * np.eye(X.shape[1])).dot(X.T.dot(y))).shape)

        # Closed-form solution for the j'th ridge regression
        target_matrix[:,j] = np.squeeze(np.linalg.inv(X.T.dot(X) + _lambda * np.eye(X.shape[1])).dot(X.T.dot(y)))

def sum_squared_error(data, U, M):
    return (np.array((training - U.T.dot(M))[training.nonzero()]) ** 2).sum()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs Alternating Least Squares (ALS) to learn latent features for users and movies.')

    # parser.add_argument('infile', help='The netflix dataset subset to use.')

    parser.add_argument('--verbose', type=int, default=0, help='Print detailed progress information to the console. 0=none, 1=high-level info, 2=all details.')
    parser.add_argument('--solver', choices=['cf', 'cd'], default='cf', help='The method used to solve alternating least squares problem. cf=closed form. cd=coordinate descent.')
    parser.add_argument('--outfile', default='results.csv', help='The file where the resulting matrix will be saved.')
    
    parser.add_argument('--converge', type=float, default=1e-6, help='The convergence threshold.')
    parser.add_argument('--max_steps', type=int, default=1000, help='The maximum number of steps to iterate.')

    parser.add_argument('--features', type=int, default=20, help='The number of latent features (k).')
    parser.add_argument('--L2', type=float, default=0.1, help='The L2 penalty parameter (lambda).')
    parser.add_argument('--plot_results', default='results.pdf', help='The file where the results plot will be saved.')

    # Get the arguments from the command line
    args = parser.parse_args()

    # Load data from file
    # Part 6 - New Model
    user_dict = defaultdict(set)
    business_dict = defaultdict(set)
    user_buisness_rating = defaultdict(dict)
    user =[]
    business=[]
    rating = []
    user_d = defaultdict()
    business_d = defaultdict()

    for l in readGz("train.json.gz"):
      a,b,c = l['userID'],l['businessID'],l['rating']
      user.append(a)
      business.append(b)
      rating.append(c)
      user_d[a] =1
      business_d[b] = 1
      user_buisness_rating[a][b] = l['rating']

    for i in range(len(user)):
      a = user[i]
      b = business[i]
      user_dict[a].add(b)
      business_dict[b].add(a)
      

    user_1 = []
    for k,v in user_d.iteritems():
      user_1.append(k)
    for i in range(len(user_1)):
      user_d[user_1[i]] = i
    business_1 = []
    for k,v in business_d.iteritems():
      business_1.append(k)
    for i in range(len(business_1)):
      business_d[business_1[i]] = i

    for i in range(len(user)):
        user[i] = user_d[user[i]]
        business[i] = business_d[business[i]]


    l = []
    for i in range(len(rating)):
        l.append((rating[i],user[i],business[i]))

    l = sorted(l, key=lambda element: (element[1], element[2]))

    print l[:100]

    for i in range(len(l)):
        a,b,c = l[i]
        rating[i] = a
        user[i] = b
        business[i] = c

    print 'CSC Data Insert'
    training = csc_matrix((rating, (user, business))) # u x m
    # testing = csc_matrix((data['Rt/data'], data['Rt/ir'], data['Rt/jc'])) # u x m
    print 'CSC Data Loaded'
    # Initialize the parameters
    delta = args.converge + 1.
    cur_error = 1.
    cur_step = 0
    U = np.ones((args.features, training.shape[0])) # k x u
    M = np.ones((args.features, training.shape[1])) # k x m

    print 'latent Factor Initialized'

    training_trace = []
    # testing_trace = []

    while delta > args.converge and cur_step < args.max_steps:
        if args.verbose:
            print 'Step #{0}'.format(cur_step)

        if args.solver == 'cf':
            # Use the closed-form solution for the ridge-regression subproblems
            if args.verbose:
                print '\tFitting M'

            cf_ridge_regression(M, U, training, args.L2, verbose=args.verbose > 1)

            if args.verbose:
                print '\tFitting U'

            cf_ridge_regression(U, M, training.T, args.L2, verbose=args.verbose > 1)
        else:
            raise Exception('Unsupported solver type: {0}'.format(args.solver))

        # Track performance in terms of RMSE on both the testing and training sets
        train_error = sum_squared_error(training, U, M)
        # test_error = sum_squared_error(testing, U, M)
        training_trace.append(np.sqrt(train_error / len(training.nonzeros()[0])))
        # testing_trace.append(np.sqrt(test_error / len(testing.nonzeros()[0])))

        # Track convergence
        prev_error = cur_error
        cur_error = train_error
        delta = np.abs(prev_error - cur_error) / (prev_error + args.converge)

        if args.verbose:
            print 'Delta: {0}'.format(delta)

        # Update the step counter
        cur_step += 1

    plt.figure()
    plt.plot(np.arange(cur_step), np.array(training_trace), label='Training RMSE')
    # plt.plot(np.arange(cur_step), np.array(testing_trace), label='Testing RMSE')
    plt.savefig(args.plot_results, bbox_inches='tight')
    plt.clf()
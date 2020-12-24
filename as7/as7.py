import numpy as np
import math


# Input: the lists of prior probabilities, likelihood, and test data
# Output: list of corresponding posterior probabilities
#
def posteriorFunc(priorProb, likhd, data):
    '''
       
       Student implements the function to calculate posterior probabilities here

	'''

    alpha = 0
    posProb = []
    space = len(priorProb)

    training = len(data)
    count = 0

    for i in data:
        if (i == 1):
            count = count + 1

    for h in range(space):

        posProb.append((pow(likhd[h], count) * pow(1 - likhd[h], training - count)) * priorProb[h])
        alpha = alpha + posProb[h]

    alpha = 1 / alpha

    for h in range(space):
        posProb[h] = posProb[h] * alpha

    return posProb


# Input the lists of prior probabilites, likhd/likelihood, training data, and one test datapoint
# Output: probability that the test datapoint happens
# Note: this function will call posteriorFunc to calculate the posterior probabilites
def predictionFunc(priorProb, likhd, data, fPoint):
    '''
       
       Student implements the function to calculate predictive probability here
       
	'''

    predictProb = 0
    postProb = posteriorFunc(priorProb, likhd, data)
    space = len(priorProb)

    for h in range(space):
        predictProb = predictProb + (likhd[h] * postProb[h])

    if (fPoint == 0):
        predictProb = 1 - predictProb

    return predictProb

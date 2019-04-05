from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
from functools import reduce
import random
from scipy.special import logsumexp

# dataDir = '/u/cs401/A3/data/'
dataDir = '../data/'


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


def precomputeStep(m, myTheta):
    mu = myTheta.mu[m]
    sigma = myTheta.Sigma[m]

    d = myTheta.Sigma.shape[1]

    result = 0

    result += reduce(lambda x, y: x + y, list(map(lambda n: pow(mu[n], 2) / (2 * pow(sigma[n], 2)), range(d))))
    result += d / 2 * np.log(2 * np.pi)
    result += 1 / 2 * np.log(reduce(lambda x, y: x * y, list(map(lambda x: pow(sigma[x], 2), range(d)))))

    return -result


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''

    sigma = myTheta.Sigma[m]
    mu = myTheta.mu[m]

    result = 0

    for n in range(len(x)):
        result -= 1 / 2 * pow(x[n], 2) * pow(sigma[n], -2)
        result += mu[n] * x[n] * pow(sigma[n], -2)

        if len(preComputedForM) < 1:
            result -= pow(mu[n], 2) / (2 * pow(sigma[n], 2))

    if len(preComputedForM) < 1:
        result + precomputeStep(m, myTheta)
    else:
        result + preComputedForM[m]

    return result


def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''

    omega = myTheta.omega
    M = len(omega)

    preComputedForM = list(map(lambda a: precomputeStep(a, myTheta), range(M)))

    prop = np.log(omega[m, 0]) + log_b_m_x(m, x, myTheta, preComputedForM)

    p_sum = logsumexp(list(map(lambda a: np.log(omega[a, 0]) + log_b_m_x(a, x, myTheta, preComputedForM), range(M))))

    return prop - p_sum


def logLik(log_Bs, myTheta):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''

    omega = myTheta.omega

    result = 0

    for t in range(log_Bs.shape[1]):
        result += logsumexp(list(map(lambda a: np.log(omega[a, 0]) + log_Bs[a, t], range(log_Bs.shape[0]))))

    return result


def compute_P(X, log_Bs, myTheta):
    log_Ps = np.zeros((M, len(X)))
    omega = myTheta.omega

    print(omega)

    for t in range(len(X)):
        sum = logsumexp(list(map(lambda m: np.log(omega[m, 0]) + log_Bs[m, t], range(M))))

        for m in range(len(omega)):
            log_Ps[m, t] = np.log(omega[m, 0]) + log_Bs[m, t] - sum

    return log_Ps


def compute_B(X, myTheta):
    M = len(myTheta.omega)
    log_Bs = np.zeros((M, len(X)))
    preComputedForM = list(map(lambda x: precomputeStep(x, myTheta), range(M)))

    for m in range(M):
        for t in range(len(X)):
            log_Bs[m, t] = log_b_m_x(m, X[t], myTheta, preComputedForM)

    return log_Bs


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta(speaker, M, X.shape[1])

    i = 0
    pre_L = float("-inf")
    improvement = float("inf")

    T = len(X)

    for m in range(M):
        myTheta.mu[m] = X[random.randint(0, T)]
        myTheta.Sigma[m] = np.ones((len(myTheta.Sigma[m])))
        myTheta.omega[m] = [1 / M]

    while i <= maxIter and improvement >= epsilon:
        print('iteration ' + str(i))

        log_Bs = compute_B(X, myTheta)
        log_Ps = compute_P(X, log_Bs, myTheta)

        L = logLik(log_Bs, myTheta)

        for m in range(M):
            sum_Ps = np.exp(logsumexp(log_Ps[m]))

            myTheta.omega[m, 0] = sum_Ps / T
            myTheta.mu[m] = np.divide(np.dot(log_Ps[m], X), sum_Ps)
            myTheta.Sigma[m] = np.divide(np.dot(log_Ps[m], np.square(X)), sum_Ps) - np.square(myTheta.mu[m])

            improvement = L - pre_L
            pre_L = L
            i = i + 1

    return myTheta


def test(mfcc, correctID, models, k=5):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1

    logLikelihood = float('-inf')

    for i in range(len(models)):
        log_Bs = compute_B(mfcc, models[i])

        log_Like = logLik(log_Bs, models[i])

        if logLikelihood < log_Like:
            bestModel = i
            logLikelihood = log_Like


    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate 
    numCorrect = 0;
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)

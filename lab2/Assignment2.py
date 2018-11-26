# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 15:53:28 2017

@author: eri
"""

import pickle
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt

def LoadBatch (file):
    X = np.array;
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data'].T/255
    y = dict[b'labels'];# list of labels as int
    Y = np.zeros((10, X.shape[1]))
    for i in range(len(y)):
        Y[y[i], i ] = 1
    return X, Y, y
    
def initP (nLayer, nNodes, dimen):
    b= []
    W=[]
    W.append(np.random.normal(0, 0.001, (nNodes[0], dimen)))
    b.append(np.zeros((nNodes[0],1)))
    for i in range(nLayer-1):
        W.append(np.random.normal(0, 0.001, (nNodes[i+1],nNodes[i])))
        b.append(np.zeros((nNodes[i+1],1)))

    return W , b
    
def EvaluateClassifier(X, W, b):
    X_l = []
    for i in range(len(W)):
        if dropout == "on" and training == "on":
            print ("dropout, training")
            DropM= np.random.choice([0, 1], size=(X.shape[0],X.shape[1]), p=[0.1, 0.9])
            X = X*DropM
        X_l.append(X)
        s1 = W[i].dot(X) + b[i]
        if method == "Relu":
            X = np.maximum(np.zeros((s1.shape[0], s1.shape[1])), s1)
        elif method == "LeakyRelu":
            X = np.maximum(0.01*s1, s1)
    denom = np.dot(np.ones((1,W[len(W)-1].shape[0])), np.exp(s1))
    p = np.exp(s1)/denom # devide each column of s with a vector element of denom
    return p, X_l 
    
def ComputeCost(X, Y, W, b, lamda):
    W2Sum = 0
    for i in range(len(W)):
        W2 = np.power(W[i], 2)
        W2Sum += W2.sum()
    P, H  = EvaluateClassifier(X, W, b)
    a = np.diag(np.dot(Y.T, P)).reshape((1,X.shape[1])) #create a vector 1xd of probs
    logP = -np.log(a)
    logPSum = logP.sum()
    loss = (1/X.shape[1])*logPSum
    cost = (1/X.shape[1])*logPSum + lamda*W2Sum
    return loss, cost  
    
    
def ComputeAccuracy(X, y, W, b):
    P, H = EvaluateClassifier(X, W, b)
    PArgMax = np.argmax(P, axis=0)
    MisClas = PArgMax - y # array (10000,). The correctly classified returns 0 the others nonzero
    sumMisClas = np.count_nonzero(MisClas)
    accuracy = 1-sumMisClas/X.shape[1]
    return accuracy
       
    
def ComputeGradients(X, Y, P, W_in, lamda):
    DLb=[]
    DLw=[]
    DJw=[]
    DJb=[]

    W = np.copy(W_in)
    
    g = (P-Y)
    for i in reversed(range(len(W))):#-1, -1, -1):
        DLb.append(np.zeros((W[i].shape[0],1)))
        DLw.append(np.zeros((W[i].shape[0],W[i].shape[1])))

        DLb[len(W)-1-i] = np.sum(g, axis=1).reshape((W[i].shape[0],1)) 
        DLw[len(W)-1-i] = np.dot(g, X[i].T) # it can be proven that it is the same as DLw[0] += np.dot(g[:,i], X[i].T[i,:]);
        DLb[len(W)-1-i] = DLb[len(W)-1-i]/X[0].shape[1]
        DLw[len(W)-1-i] = DLw[len(W)-1-i]/X[0].shape[1]
        DJw.append(DLw[len(W)-1-i] + 2*lamda*W[i])
        DJb.append(DLb[len(W)-1-i])
        #propagate gradients
        g = np.dot(g.T,W[i])
        sC = np.copy(X[i])
        if method == "Relu":
            sC[sC>0] = 1
        elif method == "LeakyRelu":
            sC[sC>0] = 1
            sC[sC==0] = 0.01
        g = np.multiply(g,sC.T) # instead of taking the diagonal have 2 nxnNodes matrices and multiply element-wise row by row
        g = g.T

    return DJw, DJb
    
    
    
def ComputeGradsNumSlow(X, Y, W, b, lamda, h = 1e-7):
    DW=[]
    Db=[]


    for k in range(len(W)):
        DWnum = np.zeros((W[k].shape[0], W[k].shape[1]))
        Dbnum = np.zeros((W[k].shape[0], 1))
        for i in range(b[k].shape[0]):
            b_try = np.copy(b)      
            b_try[k][i] -= h       
            l1, c1 = ComputeCost(X, Y, W, b_try, lamda)
            b_try[k][i] += h
            b_try = np.copy(b)
            b_try[k][i] += h
            l2, c2 = ComputeCost(X, Y, W, b_try, lamda)
            b_try[k][i] -= h
            
            Dbnum[i] = (c2-c1) / (2 * h)
            
        Db.append(Dbnum)
            
        for i in range(W[k].shape[0]):
            for j in range(W[k].shape[1]):
                W_try = np.copy(W)
                W_try[k][i][j] -= h
                l1, c1 = ComputeCost(X, Y, W_try, b, lamda)
                W_try[k][i][j] += h
                W_try = np.copy(W)
                W_try[k][i][j] += h
                l2, c2 = ComputeCost(X, Y, W_try, b, lamda)
                W_try[k][i][j] -= h
                DWnum[i][j] = (c2-c1) / (2*h)
        
        DW.append(DWnum)

    return  DW, Db
    
    
    
    
def MiniBatchGD(X, Y, y, X_val, Y_val, y_val, W, b, eta, lamda, n_batch, n_epochs, eta_decaying, rho):
    loss = []
    cost = []
    acc = []

    loss_val = []
    cost_val = []
    acc_val = []

    listW = []
    listb = []

    Vw = []
    Vb = []
    global training
    
    for i in range(len(W)):
        Vb.append(np.zeros((W[i].shape[0],1)))
        Vw.append(np.zeros((W[i].shape[0],W[i].shape[1])))
       
    for k in range(n_epochs):
        print (k)
        training = "on"
        for i in range(0,(X.shape[1]//n_batch)):
            i_start = i*n_batch
            i_end = (i+1)*n_batch 		
            Xbatch = X[:, i_start:i_end]
            Ybatch = Y[:, i_start:i_end]
            P, H = EvaluateClassifier(Xbatch, W, b)
            Dw , Db = ComputeGradients(H, Ybatch, P, W, lamda) 
            for j in range(len(W)):
                Vb[j] = rho*Vb[j] - eta*Db[len(W)-1-j]
                Vw[j] = rho*Vw[j] - eta*Dw[len(W)-1-j]
                W[j] = W[j] + Vw[j]
                b[j] = b[j] + Vb[j]

        training ="off"
        l, c = ComputeCost(X, Y, W, b, lamda)
        acc.append(ComputeAccuracy(X, y, W, b))
        loss.append(l)
        cost.append(c)
        if l>3*loss[0]:
            break;
        l_val, c_val = ComputeCost(X_val, Y_val, W, b, lamda)
        acc_val.append(ComputeAccuracy(X_val, y_val, W, b))
        loss_val.append(l_val)
        cost_val.append (c_val)        
        listW.append(W)
        listb.append(b)
        if eta_decaying == 'on':
            eta =eta*0.9
    
    idxMax = np.argmax(acc_val)
    print (loss[len(loss)-1])
    print(np.min(loss))
    print( np.max(acc_val))
    print( idxMax)
    print(eta)
    print( lamda)
    figure1, ax = plt.subplots(2, sharex=True)
    ax[0].plot(loss,label = 'training set')
    ax[0].plot(loss_val, label='validation set')
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")
    ax[1].plot(cost,label = 'training set')
    ax[1].plot(cost_val,label = 'validation set')
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("cost")
    ax[0].set_title ("Loss and Cost function vs epochs" )
    ax[0].legend()
    figure2 =plt.figure()
    plt.plot(acc,label = 'training set')
    plt.plot(acc_val, label='validation set')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs epochs")
    plt.legend()
    plt.show()

    return listW[idxMax], listb[idxMax], np.max(acc_val)
    
    
def CoarseSearch():
    
    X, Y, y = LoadBatch ('Datasets/data_batch_1')
    mean_X = np.mean(X,1).reshape(X.shape[0], 1)
    X = X - np.tile(mean_X, (1, X.shape[1]))

    X_val, Y_val, y_val = LoadBatch ('Datasets/data_batch_2')
    X_val =  X_val - np.tile(mean_X, (1, X_val.shape[1]))
                                      
    # nNodes is a list. Each value is the number of nodes at each layer
    W, b = initP(nLayer = 2, nNodes = [50, Y.shape[0]], dimen = X.shape[0])

    gridS = 8
    maxAccL= np.zeros((gridS,gridS))
    e = np.random.uniform(1,3,gridS) 
    etaL = 10**(-e)
    l = np.random.uniform(3,8,gridS)
    lamdaL = 10**(-l)
    start= default_timer() 
    k=-1
    for i in etaL:
        k +=1
        m= -1
        for j in lamdaL: 
            m+=1
            W1, maxAcc = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, np.copy(W), np.copy(b), eta=i, lamda=j, n_batch=100,  n_epochs=5,  eta_decaying="off", rho = 0.9)
            maxAccL[k,m] = maxAcc 
    (idMaxEta, idMaxLamda) = np.unravel_index(np.argmax(maxAccL), maxAccL.shape)
    print (etaL[idMaxEta])
    print (lamdaL[idMaxLamda])
    print(default_timer() - start)
    
    return etaL[idMaxEta], lamdaL[idMaxLamda]


def FineSearch(etaB, lamdaB):
    
    X, Y, y = LoadBatch ('Datasets/data_batch_1')
    mean_X = np.mean(X,1).reshape(X.shape[0], 1)
    X = X - np.tile(mean_X, (1, X.shape[1]))

    X_val, Y_val, y_val = LoadBatch ('Datasets/data_batch_2')
    X_val =  X_val - np.tile(mean_X, (1, X_val.shape[1]))
                                      
    # nNodes is a list. Each value is the number of nodes at each layer
    W, b = initP(nLayer = 2, nNodes = [50, Y.shape[0]], dimen = X.shape[0])

    gridS = 8
    maxAccL= np.zeros((gridS,gridS))
    etaL = np.random.normal(etaB,etaB/2,gridS) 
    lamdaL = np.random.normal(lamdaB,lamdaB/2,gridS)
    start= default_timer()
    k=-1
    for i in etaL:
        k +=1
        m= -1
        for j in lamdaL: 
            m+=1
            W1, maxAcc = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, np.copy(W), np.copy(b), eta=0.0228590307922, lamda=0.000788908584214, n_batch=100,  n_epochs=5,  eta_decaying="on", rho = 0.9)
            maxAccL[k,m] = maxAcc 
    (idMaxEta, idMaxLamda) = np.unravel_index(np.argmax(maxAccL), maxAccL.shape)
    print (etaL[idMaxEta])
    print (lamdaL[idMaxLamda])
    print(default_timer() - start)
    
    
    
def TrainWithAllBatches():
    # train data
    X1, Y1, y1 = LoadBatch ('Datasets/data_batch_1')
    X2, Y2, y2 = LoadBatch ('Datasets/data_batch_2')
    X3, Y3, y3 = LoadBatch ('Datasets/data_batch_3')
    X4, Y4, y4 = LoadBatch ('Datasets/data_batch_4')
    X5, Y5, y5 = LoadBatch ('Datasets/data_batch_5')
    
    X  = np.concatenate((X1,X2,X3,X4,X5[:,:9000] ), axis=1)
    Y  = np.concatenate((Y1,Y2,Y3,Y4,Y5[:,:9000] ), axis =1)
    y  = np.concatenate((y1,y2,y3,y4,y5[:9000] ))
    mean_X = np.mean(X,1).reshape(X.shape[0], 1);
    X = X - np.tile(mean_X, (1, X.shape[1]));
    
    # validation data                         
    X_val = X5[:,9000:10000];
    Y_val = Y5[:,9000:10000];
    y_val = y5[9000:10000];
    X_val =  X_val - np.tile(mean_X, (1, X_val.shape[1]));
                                      
    # nNodes is a list. Each value is the number of nodes at each layer
    W, b = initP(nLayer = 2, nNodes = [50, Y.shape[0]], dimen = X.shape[0]);

    start= default_timer(); 
    Wmax, bmax, maxAcc = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, np.copy(W), \
                                     np.copy(b), eta=0.0228590307922, lamda=0.000788908584214, n_batch=100, \
                                     n_epochs=50,  eta_decaying="on", rho = 0.9 )
    print(default_timer() - start)
    
    #test data
    X_test, Y_test, y_test = LoadBatch ('Datasets/test_batch')   
    X_test =  X_test - np.tile(mean_X, (1, X_test.shape[1]))
    
    LossTest, CostTest = ComputeCost(X_test, Y_test, Wmax, bmax, lamda = 0.000788)
    AccuracyTest = ComputeAccuracy(X_test, y_test, Wmax, bmax)
    print (LossTest)
    print (CostTest)
    print (AccuracyTest)


    
    
if __name__ == "__main__":
    
    global dropout, method
    dropout = "off"
    method = "Relu"
    
#    B_eta, B_lamda = CoarseSearch()
#    
#    FineSearch(B_eta, B_lamda)
#    
#    TrainWithAllBatches()
    
    
    
    
#    X, Y, y = LoadBatch ('Datasets/data_batch_1')
#    mean_X = np.mean(X,1).reshape(X.shape[0], 1)
#    X = X - np.tile(mean_X, (1, X.shape[1]))
#
#    X_val, Y_val, y_val = LoadBatch ('Datasets/data_batch_2')
#    X_val =  X_val - np.tile(mean_X, (1, X_val.shape[1]))
#    
#    
#    
##    X = X[:,:100].reshape((X.shape[0],100))
##    Y = Y[:,:100].reshape((10,100))
##    y = y[:100]
##    X_val = X_val[:,:100].reshape((X.shape[0],100))
##    Y_val = Y_val[:,:100].reshape((10,100))
##    y_val = y_val[:100]
#                                      
#    # nNodes is a list. Each value is the number of nodes at each layer
#    W, b = initP(nLayer = 2, nNodes = [50, Y.shape[0]], dimen = X.shape[0])
#
#    start= default_timer(); 
#    Wmax, bmax, maxAcc = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, np.copy(W), np.copy(b), eta=0.022, \
#                                     lamda=0.000788908584214, n_batch=100,  n_epochs=30,  eta_decaying="on", rho = 0.99 )
#    print(default_timer() - start)
    
    
    
    X, Y , y  = LoadBatch ('Datasets/data_batch_1')
    W, b = initP(nLayer = 2, nNodes = [50, Y.shape[0]], dimen = 100)
    
    p, X_l= EvaluateClassifier(X[:100,:10], W, b);


    #compute gradients numericaly
    grad_W_num, grad_B_num =ComputeGradsNumSlow(X[:100,:10], Y, W, b, 0, h=10**(-7))
    #compute gradients analyticaly
    grad_W_an, grad_B_an =ComputeGradients(X_l, Y[:,:10], p[:,:10], W, 0)
    #compute difference
#    print (grad_W_num[0].shape)
#    print (grad_W_num[1].shape)
    for i in reversed(range(len(W))):
#        print (i)
#        print (len(W)-1-i)
#        print (grad_W_an[len(W)-1-i].shape)
#        print (grad_W_num[i].shape)
        diffW=np.abs(grad_W_an[len(W)-1-i]-grad_W_num[i])/np.max((np.abs(grad_W_an[len(W)-1-i])+np.abs(grad_W_num[i])).clip(0))
        diffB=np.abs(grad_B_an[len(W)-1-i]-grad_B_num[i])/np.max((np.abs(grad_B_an[len(W)-1-i])+np.abs(grad_B_num[i])).clip(0))
        diff=(diffW.mean()+diffB.mean())/2.
        print('In the %s layer the Numerical to Analytical difference of the Grads is %s'%(i+1,diff))
        print('with %s being the difference in Ws'%diffW.mean())




    





   

    

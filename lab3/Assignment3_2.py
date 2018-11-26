import numpy as np
from scipy.sparse import csr_matrix
from timeit import default_timer
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import os
import pdb
import time


def ConvNet(nLen, d, K, k1, n1, k2, n2):
    f1 = np.random.randn(d, k1, n1) *np.sqrt(2.0/(nLen*d))# n1 3rd dimension(depth), d number of rows, k1 number of columns
    f2 = np.random.randn(n1, k2, n2) * np.sqrt(2.0/(n1*(nLen-k1+1)))
    W = np.random.randn(K, n2*(nLen-k1-k2+2)) * np.sqrt(2.0/(n2*(nLen-k1-k2+2)))

    return [f1, f2, W]


def readData():
    file = open('ascii_names.txt', 'r')
    lines = file.readlines()
    file.close()

    y = []
    allNames = []

    for line in lines:
        l = line.strip()
        temp = l.split()
        name = ''
        if len(temp) > 1:
            name = name.join(temp[0:-1])
        else:
            name = temp[0]
        y.append(int(temp[-1]))
        allNames.append(name)

    ys = np.asarray(y)
    all_names = np.asarray(allNames)
    n_len = len(max(all_names, key=len))  # the length of the longest word

    return ys, all_names, n_len


def createDict(ys, all_names):
    AN = ''
    AN = AN.join(all_names)

    C = list(set(AN)) # find the unique characters
    d = len(C) # number of unique characters
    n_len = len(max(all_names, key=len)) # the length of the longest word
    K = len(list(set(ys))) # number of classes

    char_to_ind = {}
    for i in range(0, len(C)):
        char_to_ind[C[i]] = i # create a dictionary (each character has its position as a key)

    return char_to_ind

def oneHotNames(char_to_ind,n_len, names):

    d = len(char_to_ind.keys())
    # one hot representation of the names X
    N = len(names)
    X = np.zeros((d*n_len,N))
    for i in range(0, N):
        en_name = np.zeros((d, n_len))
        for j in range(0, len(names[i])):
            en_name[char_to_ind[names[i][j]]][j] = 1
        X[:, i] = en_name.flatten('F')

    return X


def val_train_set(ys):

    X1 = np.load('outfile.npy')

    file = open('Validation_Inds.txt', 'r')
    lines = file.readlines()
    file.close()

    ind_val = []
    for line in lines:
        l = line.strip()
        temp = l.split()
        ind_val = [int(i)-1 for i in temp ]

    X_val = X1[:, ind_val]

    y_val = ys[ind_val]

    idx_x_columns = [i for i in range(np.shape(X1)[1]) if i not in ind_val]
    X_train = X1[:, idx_x_columns]
    y_train = ys[idx_x_columns]

    return X_train, X_val, y_train, y_val


def one_hot_labels(y, K):
    Y = np.zeros((K, len(y)))
    for i in range(len(y)):
        Y[y[i]-1, i] = 1
    return Y


def MakeMFMatrix(F, nlen):

    Vf = []
    nf = F.shape[2]
    d = F.shape[0]
    k = F.shape[1]
    for i in range(nf):
        Vf.append(F[:,:,i].flatten('F')) # Create a Vf: each row is one vectorized filter of the specific layer

    Vf = np.asarray(Vf)

    Mf = np.zeros(((nlen - k + 1)*nf, nlen*d))
    j = 0
    for i in range(0,Mf.shape[0],nf):
            Mf[i:i+nf, j:j+(d*k)] = Vf
            j = j+d

    return Mf



def MakeMXMatrix_gen(x_input, d, k, nf):

    nlen = len(x_input)//d
    x_in = x_input.reshape((d, nlen), order='F')
    Vx = []

    for i in range(nlen):
        if i <= nlen-k:
            Vx.append(x_in[:, i:i+k].flatten('F'))

    Mx_gen = np.asarray(Vx)
    return Mx_gen



def MakeMXMatrix(x_input, d, k, nf):

    nlen = len(x_input)//d
    x_in = x_input.reshape((d, nlen), order='F')
    Vx = []

    for i in range(nlen):
        if i <= nlen-k:
            Vx.append(x_in[:, i:i+k].flatten('F'))

    Vx = np.asarray(Vx)
    Mx = np.zeros(((nlen - k + 1)*nf, k*d*nf))
    l = 0
    for m in range(0, Mx.shape[0], nf):
        j = 0
        for i in range(nf):
            Mx[i+m, j:j + (d * k)] = Vx[l]
            j = j + d*k
        l = l+1

    '''much slower option'''
    # Mx = np.kron(np.eye(nf), Vx[0])
    # for i in range(1, len(Vx)):
    #     b = np.kron(np.eye(nf), Vx[i])
    #     Mx = np.vstack((Mx,b))

    return  Mx


def ComputeCost(X, Y, FWList):

    P, _= forward_pass(X, FWList)
    a = np.diag(np.dot(Y.T, P)).reshape((1, X.shape[1])) #create a vector 1xd of probs
    logP = -np.log(a)
    logPSum = logP.sum()
    loss = (1.0/X.shape[1])*logPSum
    return loss


def ComputeAccuracy(X, y, F_list):

    P, H = forward_pass(X, F_list)
    PArgMax = np.argmax(P, axis=0)
    MisClas = PArgMax - (y-1)
    sumMisClas = np.count_nonzero(MisClas)
    accuracy = 1-sumMisClas/X.shape[1]
    return accuracy


def check_s1_s2(F, x_input, n_len):

    flat_filt = []
    for i in range(F.shape[2]):
        flat_filt.append(F[:, :, i].flatten('F'))  # Create flat: each row is one vectorized filter of the specific layer


    flat_filter = np.asarray(flat_filt)

    Mf = MakeMFMatrix(F, n_len)
    Mx = MakeMXMatrix(x_input, F.shape[0], F.shape[1], F.shape[2])

    s1 = np.dot(Mf, x_input)
    s2 = np.dot(Mx, flat_filter.flatten())

    np.savetxt('s1.txt', s1)
    np.savetxt('s2.txt', s2)

    return s1


def forward_pass(X, FWList):

    nlen = X.shape[0] // FWList[0].shape[0]

    MF1 = MakeMFMatrix(FWList[0], nlen)
    MF2 = MakeMFMatrix(FWList[1], MF1.shape[0]//FWList[0].shape[2]) # devide by #Filters to calculate the correct #rows for the MF of the next layer

    #relu
    X1 = np.maximum(np.dot(MF1, X), 0)
    X2 = np.maximum(np.dot(MF2, X1), 0)
    #Fc
    S = np.dot(FWList[2], X2)
    #softmax
    denom = np.sum(np.exp(S), axis=0)
    P  = np.exp(S)/denom

    X_list  = [X, X1, X2]

    return P, X_list


def backward_pass(X_list, Y, P, FWList,Mx1F, loadMX):
    DLW = 0
    DLF1 = np.zeros((1, FWList[0].size))
    DLF2 = np.zeros((1, FWList[1].size))

    n_batch = X_list[0].shape[1]
    g = -(Y - P)
    DLW = np.dot(g, X_list[2].T)/np.float(n_batch)

    # propagate to 2nd convol from FC
    g = np.dot(FWList[2].T, g)
    sC = np.copy(X_list[2])
    sC[sC > 0] = 1
    g = np.multiply(g, sC)

    # compute gradients w.r.t 2nd convol
    for i in range(n_batch):
        xi = X_list[1][:,i]
        ## genaralized Mx
        Mx = MakeMXMatrix_gen(xi, FWList[1].shape[0], FWList[1].shape[1], FWList[1].shape[2])
        v = np.dot(Mx.T, g[:, i].reshape(g.shape[0]//FWList[1].shape[2], FWList[1].shape[2]))
        v = v.flatten('F')
        ## not generalized Mx
        # Mx = MakeMXMatrix(xi, FWList[1].shape[0], FWList[1].shape[1], FWList[1].shape[2])
        # v = np.dot(g[:, i].reshape(g.shape[0],1).T, Mx)

        DLF2 = DLF2 + v/n_batch

    DLF2 = DLF2.reshape(FWList[1].shape[0], FWList[1].shape[1], FWList[1].shape[2], order='F')


    #propagate to 1st convol from 2nd convol
    MF2 = MakeMFMatrix(FWList[1], X_list[1].shape[0]//FWList[0].shape[2])
    g = np.copy(np.dot(MF2.T, g))
    s = np.copy(X_list[1])
    s[s > 0] = 1
    g = np.multiply(g, s)

    # compute gradients w.r.t 1st convol
    for i in range(n_batch):
        if loadMX:
            Mx = Mx1F[i]
        else:
            xi = X_list[0][:, i]
            Mx = MakeMXMatrix(xi, FWList[0].shape[0], FWList[0].shape[1], FWList[0].shape[2])
        v = np.dot(g[:, i].reshape(g.shape[0],1).T, Mx)#.reshape(Mx.shape[1], 1)

        DLF1 = DLF1 + v/n_batch

    DLF1 = DLF1.reshape(FWList[0].shape[0], FWList[0].shape[1], FWList[0].shape[2], order='F')

    return [DLF1, DLF2, DLW]


def NumericalGradients(X, Y, F_list, h):
    Gs = []

    tryF_list = F_list
    for l in range(len(F_list)-1):
        G = np.zeros((np.shape(F_list[l])))
        it = np.nditer(F_list[l], flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            iF = it.multi_index
            old_value = tryF_list[l][iF]
            tryF_list[l][iF] = old_value - h  # use original value
            l1 = ComputeCost(X, Y, tryF_list)
            tryF_list[l][iF] = old_value + h  # use original value
            l2 = ComputeCost(X, Y, tryF_list)
            G[iF] = (l2 - l1) / (2 * h)
            tryF_list[l][iF] = old_value  # restore original value
            it.iternext()
        Gs.append(G)


    G = np.zeros((np.shape(F_list[2])))
    it = np.nditer(F_list[2], flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        iW = it.multi_index
        old_value = tryF_list[2][iW]
        tryF_list[2][iW] = old_value - h
        l1 = ComputeCost(X, Y, tryF_list)
        tryF_list[2][iW] = old_value + h
        l2 = ComputeCost(X, Y, tryF_list)
        G[iW] = (l2 - l1) / (2 * h)
        tryF_list[2][iW] = old_value
        it.iternext()

    Gs.append(G)

    return Gs


def MiniBatchGD(X, Y, y, X_val, Y_val, y_val, F_list, Mx, K, n_batch, eta, rho, n_updates, balanced, momentum, loadedMx):
    loss = []
    loss_val = []
    acc = []
    acc_val = []
    listW = []
    n = 0
    count = 0
    mom  = []
    for F in F_list:
        mom.append(np.zeros(F.shape))

    smallest = np.inf # number of data in the smaller class
    for i in range(K):
        n_class = np.size(np.argwhere((y-1)==i))
        if n_class < smallest:
            smallest = n_class


    from tqdm import tqdm
    pbar = tqdm(total = n_updates)
    while n < n_updates:
        print(n)

        if balanced:
            ind_B = balanced_data(y, K, smallest)
            X_t = X[:, ind_B]
            Y_t = Y[:,ind_B]
            y_t = y[ind_B]
            Mx_t = [Mx[x] for x in ind_B]
        else:
            X_t = X
            Y_t = Y
            y_t = y
            Mx_t = Mx

        for i in range(1, X_t.shape[1] // n_batch):
            i_start = (i - 1) * n_batch
            i_end = i * n_batch
            Xbatch = X_t[:, i_start:i_end]
            Ybatch = Y_t[:, i_start:i_end]

            # call forward pass
            Mx_batch = Mx_t[i_start:i_end]
            P, XList = forward_pass(Xbatch, F_list)
            G_list = backward_pass(XList, Ybatch, P, F_list,Mx_batch, loadedMx)

            if momentum:
                for j in range(len(G_list)):
                    mom[j] = rho*mom[j] + eta*G_list[j]
                    F_list[j] = F_list[j] - mom[j]
            else:
                for j in range(len(G_list)):
                    F_list[j] = F_list[j] - eta*G_list[j]
            n += 1
        l = ComputeCost(X_t, Y_t, F_list)
        l_val = ComputeCost(X_val, Y_val, F_list)
        loss.append(l)
        loss_val.append(l_val)
        acc.append(ComputeAccuracy(X_t, y_t, F_list))
        a_val  = ComputeAccuracy(X_val, y_val, F_list)
        acc_val.append(a_val)
        listW.append(F_list)

        pbar.update(X_t.shape[1] // n_batch)
    pbar.close()


    M = confusion_M(X_val,y_val, F_list)
    np.savetxt('cnf_matrix', M)

    figure1 = plt.figure()
    plt.plot(loss, label='training set')
    plt.plot(loss_val, label='validation set')
    plt.title('loss')
    figure2 = plt.figure()
    plt.plot(acc, label='training set')
    plt.plot(acc_val, label='validation set')
    plt.title('accuracy')
    plt.legend(loc = 'upper left')
    plt.show()

    idxMaxW = np.argmax(acc_val)
    F = listW[idxMaxW]

    return F, np.max(acc_val), idxMaxW

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename + '.npz')
    a = csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
    return a

def make_sparse(X, d, k, nf):
    newpath = './data'
    try:
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    except OSError:
        print ('Error: Creating directory. ' + directory)

    for i in range(X.shape[1]):
        filename = 'sm_'+ str(i)
        name = os.path.join(newpath,filename)
        xi = X[:, i]
        Mx = MakeMXMatrix(xi, d, k, nf)
        SM = csr_matrix(Mx)
        save_sparse_csr(name, SM)



def balanced_data(y, K, smallest):

    ind_b = []
    for i in range(K):
        ind_class_i = np.argwhere((y-1)==i).reshape(-1)
        ind_b.extend(np.random.choice(ind_class_i, smallest, replace = False))

    np.random.shuffle(ind_b)

    return ind_b

def confusion_M(X,y, fList):
    P,_ = forward_pass(X, fList)

    PArgMax = np.argmax(P, axis=0)
    CM = np.zeros((P.shape[0], P.shape[0]), dtype=int)
    for i in range(len(y)):
        CM[y[i]-1, PArgMax[i]] += 1

    return CM


def search_size():

    np.random.seed(100)
    ys, all_names, n_len = readData()
    K = len(list(set(ys)))  # number of classes
    X, X_val, y, y_val = val_train_set(ys)
    Y = one_hot_labels(y, K)
    Y_val = one_hot_labels(y_val, K)
    d = X.shape[0]//n_len

    Mx = []

    gSA = np.empty([5,9])
    start = time.time()
    ki = 0
    for k in range(5, 18, 3):
        ni = 0
        for n in range(10, 51, 5):
            k1  = k
            k2 = (19 - k1)//2
            n1 = n
            n2 = n
            F_W_list = ConvNet(n_len, d, K, k1, n1, k2, n2)
            maxW, maxAcc = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, F_W_list, Mx, K,
                n_batch = 100, eta = 0.001, rho = 0.9, n_updates = 500, balanced = True, momentum = True, loadedMx = False )
            gSA[ki,ni] = maxAcc
            ni +=1
        ki +=1
    end = time.time()
    elapsed = end - start
    print(elapsed)

    plt.imshow(gSA, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('number of filters')
    plt.ylabel('size of filter')
    plt.colorbar()
    plt.yticks(np.arange(5), [5, 8 ,11, 14, 17])
    plt.xticks(np.arange(9), [10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.show()


def search_param():
    np.random.seed(100)
    ys, all_names, n_len = readData()
    K = len(list(set(ys)))  # number of classes
    X, X_val, y, y_val = val_train_set(ys)
    Y = one_hot_labels(y, K)
    Y_val = one_hot_labels(y_val, K)
    d = X.shape[0]//n_len
    k1 = 8
    n1 = 20
    k2 = 5
    n2 = 20
    F_W_list = ConvNet(n_len, d, K, k1, n1, k2, n2)

    # load sparse matrix
    Mx = []
    for j in range(0,X.shape[1]):
        filename = './data/sm_'+ str(j)
        array  = load_sparse_csr(filename)
        Mx.append(array.toarray())

    gSA = np.empty([10,11])
    start = time.time()
    ki = 0
    for h in np.arange(0.0001, 0.01, step=0.001):
        ni = 0
        for r in np.arange(0.7, 0.9, step=0.02):
            maxW, maxAcc = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, F_W_list, Mx, K,
                n_batch = 100, eta = h, rho = r, n_updates = 5000, balanced = True, momentum = True, loadedMx = True )
            gSA[ki,ni] = maxAcc
            ni +=1
        ki +=1
    end = time.time()
    elapsed = end - start
    print(elapsed)

    plt.imshow(gSA, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('rho')
    plt.ylabel('eta')
    plt.colorbar()
    plt.xticks(np.arange(11), [0.7,0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.9])
    plt.yticks(np.arange(10), [0.0001, 0.0011, 0.0021, 0.0031, 0.0041, 0.0051, 0.0061, 0.0071, 0.0081, 0.0091])
    plt.show()





if __name__ == '__main__':

    '''
        Compare the convolutional matrices with
        the one given in the DebugInfo.mat file
    '''

    # mat = scipy.io.loadmat('DebugInfo.mat')
    # F = (mat['F'])
    # x_input = mat['x_input']
    # n_len = 19
    #
    # s = check_s1_s2(F, x_input, n_len)
    # np.savetxt('vecS', mat['vecS'])
    #
    # print (np.sum(s- mat['vecS']))



    '''
        check the correct computation of convolutions using both methods.
        check_s1_s2() function calls the MakeMXMatrix() and MakeMFMatrix()
        computes the convolutions for the first input of training data and
        stores the result in txt files s1.txt and s2.txt
     '''

    # X = np.load('outfile.npy')
    # x_input = X[:, 10953]
    # n_len = 19
    # F = ConvNet(n_len, d = 54, k1=2, n1=3, k2=2, n2=1, K=10)
    # check_s1_s2(F[0], x_input, n_len)

    '''
    main implementation of the neural network
    '''


    np.random.seed(100)
    ys, all_names, n_len = readData()
    Dict = createDict(ys, all_names)

    test_names  =[ 'Ichtiaroglou', 'Sager','Kassiou','Giramondi', 'Coppens']
    test_names_OH = oneHotNames(Dict,n_len, test_names)


    # # ------create the one hot representation of the training names and save them to output file ------
    # # !!!!!!!!!!!!!! ALREADY DONE!!!!!!!!!!!
    # Xnames = oneHotNames(Dict,n_len, all_names)
    # np.save('outfile', X)

    K = len(list(set(ys)))  # number of classes
    X, X_val, y, y_val = val_train_set(ys)
    Y = one_hot_labels(y, K)
    Y_val = one_hot_labels(y_val, K)
    d = X.shape[0]//n_len
    k1 = 8
    n1 = 20
    k2 = 5
    n2 = 20
    F_W_list = ConvNet(n_len, d, K, k1, n1, k2, n2)


    ##------ call make_sparse() whenever you change the size or the number of filters per layer -----
    # make_sparse(X, d, k1, n1)

    ## load sparse matrix
    Mx = []
    for j in range(0,X.shape[1]):
        filename = './data/sm_'+ str(j)
        array  = load_sparse_csr(filename)
        Mx.append(array.toarray())


    maxW, maxAcc, idW = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, F_W_list, Mx, K,
        n_batch = 100, eta = 0.001, rho = 0.9, n_updates = 20000, balanced = True, momentum = True, loadedMx = True )


    P, _ = forward_pass(test_names_OH, maxW)

    print (maxAcc)
    print (idW)
    print (P)


    '''
    test gradients
    '''
    # np.random.seed(100)
    # ys, all_names, n_len = readData()
    # K = len(list(set(ys)))  # number of classes
    # X, X_val, y, y_val = val_train_set(ys)
    # Y = one_hot_labels(y, K)
    # Y_val = one_hot_labels(y_val, K)
    # d = X.shape[0]//n_len
    # k1 = 8
    # n1 = 20
    # k2 = 5
    # n2 = 20
    # F_W_list = ConvNet(n_len, d, K, k1, n1, k2, n2)
    #
    # # load sparse matrix
    # Mx = []
    # for j in range(0,X.shape[1]):
    #     filename = './data/sm_'+ str(j)
    #     array  = load_sparse_csr(filename)
    #     Mx.append(array.toarray())
    #
    # # compute gradients numericaly
    # NumG= NumericalGradients(X[:, :100], Y[:, :100], F_W_list, h=1e-6)
    #
    # # compute gradients analyticaly
    # P, Xlist = forward_pass(X[:,:100], F_W_list)
    # AnG = backward_pass(Xlist, Y[:, :100], P, F_W_list,Mx[:100], loadMX = True)
    #
    # dif = 0
    # for i in reversed(range(len(NumG))):
    #
    #     diffW = np.abs(AnG[i] - NumG[i]) / np.max(
    #         (np.abs(AnG[i]) + np.abs(NumG[i])).clip(0))
    #     diff = diffW.mean()
    #     dif += diff/len(NumG)
    #     print('In the %s layer the Numerical to Analytical difference of the Grads is %s' % (i + 1, diff))
    # print( dif )

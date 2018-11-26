import numpy as np
import matplotlib.pyplot as plt
import copy


class RNN():

    def __init__(self, K, m):
        self.K = K
        self.m = m
        self.b = np.zeros((m, 1))
        self.c = np.zeros((K, 1))
        self.h = np.zeros((m, 1))
        self.U = np.random.randn(m, K) * 0.01
        self.W = np.random.randn(m, m) * 0.01
        self.V = np.random.randn(K, m) * 0.01


def readData():
    file = open('goblet_book.txt', 'r')
    lines = file.readlines()
    file.close()

    book_data = []
    for line in lines:
        book_data.extend(line)

    return book_data


def createDict(bookData):
    AN = ''
    AN = AN.join(bookData)

    book_chars = list(set(AN))  # find the unique characters
    K = len(book_chars)  # number of unique characters

    char_to_ind = {}
    for i in range(0, K):
        char_to_ind[book_chars[i]] = i  # create a dictionary (each character has its position as a key)

    ind_to_char = {}
    for i in range(0, K):
        ind_to_char[i] = book_chars[i]  # create a dictionary (each character has its position as a key)

    return char_to_ind, ind_to_char


def oneHotBook(char_to_ind, bookData):
    K = len(char_to_ind.keys())
    # one hot representation of the names X
    N = len(bookData)
    X = np.zeros((K, N))
    for i in range(0, N):
        X[char_to_ind[bookData[i]]][i] = 1

    return X


def synthesizeText(RNN, x0, n):
    W = RNN.W
    U = RNN.U
    b = RNN.b
    V = RNN.V
    c = RNN.c
    K = RNN.K

    h = RNN.h
    x = x0
    y = []

    for i in range(0, n):
        a = np.dot(W, h) + np.dot(U, x) + b
        h = np.tanh(a)
        o = np.dot(V, h) + c
        p = np.exp(o) / np.sum(np.exp(o), axis=0)

        cp = np.cumsum(p, axis=0)
        a = np.random.rand()
        ixs = np.nonzero(cp - a > 0)
        ii = ixs[0][0]

        x = np.zeros((K, 1))
        x[ii, 0] = 1  # one hot representaion of the next input, which is also the output
        y.append(x)

    return y


def oh_to_characters(y, ind_to_char):
    seq = ''
    for i in range(len(y)):
        ind = np.where(y[i] != 0)
        seq += ind_to_char[ind[0][0]]

    return seq

def forward_pass(X, Y, RNN):
    W = RNN.W
    U = RNN.U
    b = RNN.b
    V = RNN.V
    c = RNN.c
    ht = RNN.h

    P = {}
    H = {}
    H[-1] = ht
    loss = 0

    for t in range(X.shape[1]):
        Xt = X[:, t].reshape(X.shape[0], 1)
        a_t = np.dot(U, Xt) + np.dot(W, H[t-1]) + b
        H[t] = np.tanh(a_t)
        o_t = np.dot(V, H[t]) + c
        P[t] = np.exp(o_t) / np.sum(np.exp(o_t))

        loss += -np.log(np.dot(Y[:, t], P[t]))

    return P, H, loss



def backward_pass(P, H, X, Y, RNN):
    dW, dV, dU, db, dc = np.zeros_like(RNN.W), np.zeros_like(RNN.V), np.zeros_like(RNN.U), np.zeros_like(
        RNN.b), np.zeros_like(RNN.c)

    da = np.zeros_like(H[0])

    for t in reversed(range(X.shape[1])):
        Yt = Y[:,t].reshape(Y.shape[0], 1)
        Xt = X[:,t].reshape(X.shape[0], 1)
        g = -(Yt - P[t])

        dV += np.dot(g, H[t].T)
        dc += g

        dh = (np.dot(RNN.V.T, g) + np.dot(RNN.W.T, da))
        da = dh * (1 - H[t] ** 2)

        dW += np.dot(da, H[t-1].T)
        db += da

        dU += np.dot(da, Xt.T)

        dU = np.maximum(np.minimum(dU, 5), -5)
        dW = np.maximum(np.minimum(dW, 5), -5)
        dV = np.maximum(np.minimum(dV, 5), -5)
        db = np.maximum(np.minimum(db, 5), -5)
        dc = np.maximum(np.minimum(dc, 5), -5)

    return dW, dV, dU, db, dc, H[-1]


def check_grad( X_hot, Y_hot, grads, RNN, h=1e-4):

    ''' Compute grads numerically '''
    grad_U_num = np.zeros(RNN.U.shape)
    grad_W_num = np.zeros(RNN.W.shape)
    grad_V_num = np.zeros(RNN.V.shape)
    grad_b_num = np.zeros(RNN.b.shape)
    grad_c_num = np.zeros(RNN.c.shape)

    print("Compute Grads Numerically")
    for i in range(len(RNN.b)):
        RNN_try = copy.deepcopy(RNN)
        RNN_try.b[i] -= h
        _, _, c1 = forward_pass(X_hot, Y_hot, RNN_try)
        RNN_try = copy.deepcopy(RNN)
        RNN_try.b[i] += h
        _, _, c2 = forward_pass(X_hot, Y_hot, RNN_try)
        grad_b_num[i] = (c2 - c1) / (2 * h)

    for i in range(len(RNN.c)):
        RNN_try = copy.deepcopy(RNN)
        RNN_try.c[i] -= h
        _, _, c1 = forward_pass(X_hot, Y_hot, RNN_try)
        RNN_try = copy.deepcopy(RNN)
        RNN_try.c[i] += h
        _, _, c2 = forward_pass(X_hot, Y_hot, RNN_try)
        grad_c_num[i] = (c2 - c1) / (2 * h)

    for i in range(RNN.U.shape[0]):
        for j in range(RNN.U.shape[1]):
            RNN_try = copy.deepcopy(RNN)
            RNN_try.U[i, j] -= h
            _, _, c1 = forward_pass(X_hot, Y_hot, RNN_try)
            RNN_try = copy.deepcopy(RNN)
            RNN_try.U[i,j] += h
            _, _, c2 = forward_pass(X_hot, Y_hot, RNN_try)
            grad_U_num[i, j] = (c2 - c1) / (2 * h)

    for i in range(RNN.W.shape[0]):
        for j in range(RNN.W.shape[1]):
            RNN_try = copy.deepcopy(RNN)
            RNN_try.W[i, j] -= h
            _, _, c1 = forward_pass(X_hot, Y_hot, RNN_try)
            RNN_try = copy.deepcopy(RNN)
            RNN_try.W[i, j] += h
            _, _, c2 = forward_pass(X_hot, Y_hot, RNN_try)
            grad_W_num[i, j] = (c2 - c1) / (2 * h)

    for i in range(RNN.V.shape[0]):
        for j in range(RNN.V.shape[1]):
            RNN_try = copy.deepcopy(RNN)
            RNN_try.V[i, j] -= h
            _, _, c1 = forward_pass(X_hot, Y_hot, RNN_try)
            RNN_try = copy.deepcopy(RNN)
            RNN_try.V[i, j] += h
            _, _, c2 = forward_pass(X_hot, Y_hot, RNN_try)
            grad_V_num[i, j] = (c2 - c1) / (2 * h)

    ''' Check '''
    res_U = np.average(np.absolute(grads[2] - grad_U_num)) / np.amax(np.absolute(grads[2]) + np.absolute(grad_U_num))
    res_W = np.average(np.absolute(grads[0] - grad_W_num)) / np.amax(np.absolute(grads[0]) + np.absolute(grad_W_num))
    res_V = np.average(np.absolute(grads[1] - grad_V_num)) / np.amax(np.absolute(grads[1]) + np.absolute(grad_V_num))

    res_b = np.average(np.absolute(grads[3] - grad_b_num)) / np.amax(np.absolute(grads[3]) + np.absolute(grad_b_num))
    res_c = np.average(np.absolute(grads[4] - grad_c_num)) / np.amax(np.absolute(grads[4]) + np.absolute(grad_c_num))

    res = {res_U: 'U', res_W: 'W', res_V: 'V', res_b: 'b', res_c: 'c'}
    for r in res:
        if r < 1e-4:
            print("error for " + res[r] + ": =====>", r)
            print("Accepted!", '\n')
        else:
            print("error for " + res[r] + ": =====>", r)
            print("Warning...!  The absolute difference should be around the order 1e-6.", '\n')


def gradient_check():
    book_data = readData()
    char_to_ind, ind_to_char = createDict(book_data)
    K = len(char_to_ind.keys())
    RNN_p = RNN(K, 100)
    seq_length = 20

    X_chars = book_data[0:seq_length]
    Y_chars = book_data[1:seq_length + 1]

    X = oneHotBook(char_to_ind, X_chars)
    Y = oneHotBook(char_to_ind, Y_chars)

    P, H, _ = forward_pass(X, Y, RNN_p)
    dW, dV, dU, db, dc, _ = backward_pass(P, H, X, Y, RNN_p)

    grads = [dW, dV, dU, db, dc]

    check_grad(X, Y, grads, RNN_p, h=1e-4)

def mini_batch():
    book_data = readData()
    char_to_ind, ind_to_char = createDict(book_data)
    K = len(char_to_ind.keys())
    m = 100
    RNN_p = RNN(K, m)
    seq_length = 20
    eta = 0.1

    e = 0
    iteration = 0
    smooth_loss = -np.log(1 / K) * seq_length
    smooth_lossL = []
    UL, WL, VL, bL, cL = [], [], [], [], []
    mW, mV, mU, mb, mc = np.zeros_like(RNN_p.W), np.zeros_like(RNN_p.V), np.zeros_like(RNN_p.U), np.zeros_like(
        RNN_p.b), np.zeros_like(RNN_p.c)

    while iteration < 100000:
        if iteration == 0 or e >= len(book_data) - seq_length - 1:
            e = 0
            RNN_p.h = np.zeros((m, 1))

        X_chars = book_data[e: e + seq_length]
        Y_chars = book_data[e + 1: e + 1 + seq_length]

        X = oneHotBook(char_to_ind, X_chars)
        Y = oneHotBook(char_to_ind, Y_chars)

        P, H, loss = forward_pass(X, Y, RNN_p)
        dW, dV, dU, db, dc, RNN_p.h = backward_pass(P, H, X, Y, RNN_p)

        smooth_loss = .999 * smooth_loss + .001 * loss
        smooth_lossL.append(smooth_loss)

        # if iteration % 1000 == 0:
        #     print("iteration = " + str(iteration), "loss = " + str(smooth_loss))

        if iteration % 10000 == 0:
            print("-" * 100)
            print("Synth text iteration " + str(iteration))
            y = synthesizeText(RNN_p, X[:, 0], n=200)
            text = oh_to_characters(y, ind_to_char)
            print(text)
            print("-" * 70)

        grads = [dW, dV, dU, db, dc]
        params = [RNN_p.W, RNN_p.V, RNN_p.U, RNN_p.b, RNN_p.c]
        updateP = [mW, mV, mU, mb, mc]

        for n in range(len(params)):
            updateP[n] += grads[n] ** 2
            params[n] += - (eta * grads[n]) / np.sqrt(updateP[n] + 1e-8)

        UL.append(RNN_p.U)
        WL.append(RNN_p.W)
        VL.append(RNN_p.V)
        bL.append(RNN_p.b)
        cL.append(RNN_p.c)

        e += seq_length
        iteration += 1

    best_ind = np.argmin(smooth_lossL)

    RNN_p.U = UL[best_ind]
    RNN_p.W = WL[best_ind]
    RNN_p.V = VL[best_ind]
    RNN_p.b = bL[best_ind]
    RNN_p.c = cL[best_ind]

    print("Synth text best model")
    y = synthesizeText(RNN_p, X[:, 0], n=1000)
    text = oh_to_characters(y, ind_to_char)
    print(text)
    print("-" * 100)

    figure1 = plt.figure()
    plt.plot(smooth_lossL, label='loss')
    plt.ylabel('Smooth Loss')
    plt.xlabel('Iterations')
    plt.show()


if __name__ == '__main__':

    '''Gradient check'''
    gradient_check()

    '''Train the model'''
    mini_batch()

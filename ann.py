


import os
import sys
import numpy as np
import ctypes


#ANN_DLL = ctypes.cdll.LoadLibrary(r"/home/maxim/kaggle/ann/libann.so")
ANN_DLL = ctypes.cdll.LoadLibrary(r"c:\\temp\\test_python\\ann\\ann_sse2.dll")




class ANN(object):
    def __init__(self, sizes, dor):
        self.ss = np.array(sizes, dtype=np.int32)
        self.ann = ANN_DLL.ann_create(self.ss.ctypes.data, ctypes.c_int(self.ss.shape[0]), ctypes.c_double(dor))
        self.alpha = ctypes.c_double(.004)
        self.cost = ctypes.c_double(0.)


    def partial_fit(self, X, Y, dummy=None, n_iter=1, L=0., A=.004):
        self.alpha = ctypes.c_double(A)
        R = X.shape[0] if len(X.shape) == 2 else 1
        ANN_DLL.ann_fit(ctypes.c_void_p(self.ann), X.ctypes.data, Y.ctypes.data, ctypes.c_int(R), ctypes.addressof(self.alpha), ctypes.c_double(L), ctypes.c_int(n_iter), ctypes.addressof(self.cost))

    def fit(self, X, Y):
        self.partial_fit(X, Y, n_iter=1000)


    def predict_proba(self, X):
        if type(X) == list:
            X = np.array(X, dtype=np.float64)
        R = X.shape[0] if len(X.shape) == 2 else 1
        C = self.ss[-1]
        predictions = np.array([0] * R * C, dtype=np.float64)
        ANN_DLL.ann_predict(ctypes.c_void_p(self.ann), X.ctypes.data, predictions.ctypes.data, ctypes.c_int(R))

        predictions = predictions.reshape((R, C))

        if C == 1:
            res = np.zeros((R, 2), dtype=np.float64)
            for i,v in enumerate(predictions):
                res[i,0] = 1. - v[0]
                res[i,1] = v[0]
            return res

        return predictions



    def get_weights(self):
        ww_size = 0
        bb_size = 0

        for l in range(1, self.ss.shape[0]):
            bb_size += self.ss[-l]
            ww_size += self.ss[-l] * self.ss[-l-1]
            
        ww = np.zeros((ww_size,), dtype=np.float64)
        bb = np.zeros((bb_size,), dtype=np.float64)

        ANN_DLL.ann_get_weights(ctypes.c_void_p(self.ann), ww.ctypes.data, bb.ctypes.data)

        return ww, bb


    def set_weights(self, ww, bb):
        ww_size = ww.shape[0]
        bb_size = bb.shape[0]

        ANN_DLL.ann_set_weights(ctypes.c_void_p(self.ann), ww.ctypes.data, ctypes.c_int(ww_size), bb.ctypes.data, ctypes.c_int(bb_size))



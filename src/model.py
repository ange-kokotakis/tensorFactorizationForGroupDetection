import numpy as np
from tqdm import tqdm
from gentest import gen_contact_tensors

class model:
    def __init__(self, path, R, timestep, l = 1, eps = 1e-15, showNormsHistory = True):
        self.R = R
        self.timestep = timestep
        self.l = l
        self.eps = eps
        self.showNormsHistory = showNormsHistory
        self.Y = gen_contact_tensors(path, timestep)
        self.I = self.Y.shape[1]
        self.K = self.Y.shape[0]
        self.U = np.random.rand(self.I, R) + eps
        self.V = np.random.rand(self.I, R) + eps
        self.W = np.random.rand(self.K, R) + eps
        self.S = self._calculS()
        self.norml0 = self._costFunction()
        self.normUV = np.linalg.norm(self.U - self.V)**2
        self.listNorml0 = np.array([])
        self.listNormUV = np.array([])
        
        #Normalization of each column of U, V, W
        normsU = np.linalg.norm(self.U, axis = 0)
        self.U /= normsU
        
        normsV = np.linalg.norm(self.V, axis = 0)
        self.V/= normsV

        normsW = np.linalg.norm(self.W, axis = 0)
        self.W /= normsW

    def _calculS(self):
        S = np.zeros((self.K, self.I, self.I))
        for r in range(self.R):
            S += self.U[:, r].reshape(1, self.I, 1) * self.V[:, r].reshape(1, 1, self.I) * self.W[:, r].reshape(self.K, 1, 1)
        return S

    def _calculS2D(self, U, V, r1 = None):
        S = np.zeros((len(U), len(V)))
        for r in range(self.R):
            S += U[:, r].reshape(-1, 1) @ V[:, r].reshape(1, -1)
        if r1 != None:
            S -= U[:, r1].reshape(-1, 1) @ V[:, r1].reshape(1, -1)
        return S

    @staticmethod
    def _KhatriRao(A, B):
        '''
        Column-wise Kronecker product
        '''
        return np.repeat(A, repeats = B.shape[0], axis = 0) * np.tile(B, (A.shape[0],1))

    @staticmethod
    def _mode1Unfolding(T):
        '''
        Horizontal stacking of frontal slices
        '''
        return np.hstack(T)

    @staticmethod
    def _mode2Unfolding(T):
        '''
        Horizontal stacking of transposed frontal slices
        '''
        return np.vstack(T).T

    @staticmethod
    def _mode3Unfolding(T):
        '''
        Horizontal stacking of transposed vertical slices
        '''
        return np.hstack(np.dstack(np.dstack(T)))

    def _costFunction(self):
        return np.linalg.norm(self.Y - self.S)**2 + self.l * np.linalg.norm(self.U - self.V)**2

    def _gradU(self):
        B = self._KhatriRao(self.W, self.V).T
        positivePart = 2*(self.U @ B @ B.T + self.l * self.U)
        negativePart = np.maximum(2 * (self._mode1Unfolding(self.Y) @ B.T + self.l * self.V), self.eps)
        return positivePart, negativePart

    def _gradV(self):
        B = self._KhatriRao(self.W, self.U).T
        positivePart = 2*(self.V @ B @ B.T + self.l * self.V)
        negativePart = np.maximum(2 * (self._mode2Unfolding(self.Y) @ B.T + self.l * self.U), self.eps)
        return positivePart, negativePart

    def _gradW(self):
        B = self._KhatriRao(self.V, self.U).T
        positivePart = 2 * self.W @ B @ B.T
        negativePart = np.maximum(2 * self._mode3Unfolding(self.Y) @ B.T, self.eps)
        return positivePart, negativePart

    def _iterationMU(self, A, gradX):
        positivePart, negativePart = gradX() 
        return A * (negativePart / positivePart)

    def MU(self, nbIterations):
        for k in tqdm(range(nbIterations)):
            self.U = self._iterationMU(self.U, self._gradU)
            self.V = self._iterationMU(self.V, self._gradV)
            self.W = self._iterationMU(self.W, self._gradW)
            if self.showNormsHistory:
                self.S = self._calculS()
                self.listNorml0 = np.append(self.listNorml0, self._costFunction())
                self.listNormUV = np.append(self.listNormUV, np.linalg.norm(self.U - self.V)**2)
        self.S = self._calculS()
        self.norml0 = self._costFunction()
        self.normUV = np.linalg.norm(self.U - self.V)**2

    def HALS(self, nbIterations):
        k = 0
        while k <= nbIterations:
            for r in range(self.R):
                if k > nbIterations:
                    break
                hU = self._KhatriRao(self.W, self.V) 
                Yr = self._mode1Unfolding(self.Y) - self._calculS2D(self.U, hU, r)
                self.U[:, r] = np.maximum((1/(self.l + np.linalg.norm(hU[:, r])**2)) *(self.l *self.V[:, r] + Yr @ hU[:, r]), 0)

                hV = self._KhatriRao(self.W, self.U)
                Yr = self._mode2Unfolding(self.Y) - self._calculS2D(self.V, hV, r)
                self.V[:, r] = np.maximum((1/(self.l + np.linalg.norm(hV[:, r])**2)) *(self.l *self.U[:, r] +Yr @ hV[:, r]), 0)
            
                hW = self._KhatriRao(self.V, self.U)
                Yr = self._mode3Unfolding(self.Y) - self._calculS2D(self.W, hW, r)
                self.W[:, r] = np.maximum((1/(self.l + np.linalg.norm(hW[:, r])**2)) * Yr @ hW[:, r], 0)

                k+=1
                if self.showNormsHistory:
                    self.S = self._calculS()
                    self.listNorml0 = np.append(self.listNorml0, self._costFunction())
                    self.listNormUV = np.append(self.listNormUV, np.linalg.norm(self.U - self.V)**2)
        self.S = self._calculS()
        self.norml0 = self._costFunction()
        self.normUV = np.linalg.norm(self.U - self.V)**2    
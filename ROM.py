__all__ = [
    "ROM_Full_Data", 
    "ROM_Full_Data_Forcing",
    "ROM_Partial_Data"
]

import abc
import numpy as np
from utils import *

class ROM_Full_Data_Base(abc.ABC):

    rcond = np.finfo(float).eps

    def __init__(self, Phi, PhiTilde, PhiD, dt):

        self.Phi, self.PhiTilde, self.PhiD = Phi, PhiTilde, PhiD
        self.dt = dt
        

        self.PhiHalf = 0.5 * (self.Phi[..., :-1] + self.Phi[..., 1:])
        self.PhiTildeHalf = 0.5 * (self.PhiTilde[..., :-1] + self.PhiTilde[..., 1:])


        self.Ns, self.d, self.Nt = Phi.shape
        self.dTilde = PhiTilde.shape[1]
        self.N = self.d + self.dTilde

        self.R      = NotImplemented
        self.RTilde = NotImplemented

        self.K = NotImplemented
        self.B = NotImplemented
    
    def solveR(self):
        R = np.zeros((self.d, self.d, self.Nt-1))
        RTilde = np.zeros((self.d, self.dTilde, self.Nt-1))
        for n in range(self.Nt-1):
            F = np.hstack((self.PhiHalf[..., n], self.PhiTildeHalf[..., n]))
            X = np.linalg.pinv(F, rcond=self.rcond) @ self.PhiD[...,n]
            R[..., n], RTilde[..., n] = X[:self.d,:].T, X[self.d:,:].T
        return R, RTilde
    
    def solveConstantR(self):
        _F = np.concatenate(
            (self.PhiHalf[..., :self.Nt-1], self.PhiTildeHalf[..., :self.Nt-1]),
            axis=1
        ) 
        F = _F.transpose(2, 0, 1).reshape(-1, _F.shape[1])  
        Z = self.PhiD.transpose(2, 0, 1).reshape(-1, self.PhiD.shape[1])
        X = np.linalg.pinv(F, rcond=self.rcond) @ Z 

        R = np.zeros((self.d, self.d, self.Nt-1))
        RTilde = np.zeros((self.d, self.dTilde, self.Nt-1))

        R[...] = X[:self.d, :].T[..., None]
        RTilde[...] = X[self.d:, :].T[..., None]

        return R, RTilde
    
    def train(self, constantR=True, memorySupport=1.0):
        if constantR: 
            self.R, self.RTilde = self.solveConstantR()
        else:
            self.R, self.RTilde = self.solveR()

        self.K, self.B = self.solveKB(memorySupport)
    
    @abc.abstractmethod
    def solve(self, Phi0, PhiTilde0, dt, Nt): raise NotImplementedError


class ROM_Full_Data(ROM_Full_Data_Base):

    def __init__(self, Phi, PhiTilde, PhiD, dt, Ksolver="direct", maxiter=500, tol=1e-12):
        super().__init__(Phi, PhiTilde, PhiD, dt)

        # lsqr 
        self.Ksolver = Ksolver
        self.maxiter = maxiter
        self.tol = tol

        if self.Ksolver not in ["direct", "lsqr"]:
            raise Exception("Unknown Ksolver. Use 'direct' or 'lsqr'.")


    def solveKB(self, memorySupport):

        if memorySupport < 1 and self.Ksolver == "direct":
            raise Exception("Direct solver cannot be used with truncated memory support. Use 'lsqr'")
        
        K = np.zeros((self.d, self.d, self.PhiD.shape[-1]))
        B = np.zeros((self.d, self.dTilde, self.PhiD.shape[-1]))

        if self.Ksolver == 'lsqr':
            m_ = int(memorySupport*self.PhiD.shape[-1])
            BR = np.einsum('mpn,qpn->mqn', self.PhiHalf, self.R)
            Z = self.PhiD - BR
            K_, B, _ = lsqrKB(self.PhiHalf, self.PhiTilde[..., 0], Z, self.dt, m_, self.rcond, maxiter=self.maxiter, tol=self.tol)
            K[...,:K_.shape[-1]] = K_
            
        else:
            F = np.hstack((self.PhiHalf[..., 0], self.PhiTilde[..., 0]))
            pinvF = np.linalg.pinv(F, rcond=self.rcond)
            K = np.zeros((self.d, self.d, self.Nt-1))
            B = np.zeros((self.d, self.dTilde, self.Nt-1))

            for n in range(self.Nt-1):
                Z = self.PhiD[..., n] - self.PhiHalf[...,n] @ self.R[...,n].T      
                if n == 0:
                    X = pinvF @ Z
                    K[..., n], B[..., n] = X[:self.d,:].T/self.dt*2, X[self.d:,:].T
                else:
                    for k in range(1,n):
                        Z += -self.dt * self.PhiHalf[..., k] @ K[..., n-k].T
                    Z += -self.dt/2 * self.PhiHalf[..., n] @ K[..., 0].T
                    X = pinvF @ Z
                    K[..., n], B[..., n] = X[:self.d,:].T/self.dt, X[self.d:,:].T
        return K, B


    def solve(self, Phi0, PhiTilde0):
        sol = np.zeros((Phi0.shape[0], Phi0.shape[1], self.Nt))
        sol[..., 0] = Phi0
        for n in range(self.Nt-1):
            mat = np.eye(self.R.shape[0]) - self.dt/2*self.R[..., n].T - self.dt**2/4 * self.K[..., 0].T
            matp = np.eye(self.R.shape[0]) + self.dt/2*self.R[..., n].T + self.dt**2/4 * self.K[..., 0].T
            rhs = sol[..., n]@matp + self.dt*PhiTilde0@self.B[..., n].T 
            for k in range(n):
                 rhs += self.dt**2 * (sol[..., k]+sol[..., k+1])/2 @ self.K[..., n-k].T
            sol[..., n+1] = rhs @ np.linalg.inv(mat) 
        return sol
    

class ROM_Full_Data_Forcing(ROM_Full_Data_Base):

    def __init__(self, Phi, PhiTilde, PhiD, G, GTilde, dt):
        super().__init__(Phi, PhiTilde, PhiD, dt)
        
        self.G = G
        self.GTilde = GTilde
        self.PhiD -= G


    def solveKB(self, *arg): 
        K = np.zeros((self.d, self.d, self.PhiD.shape[-1]))
        B = np.zeros((self.d, self.dTilde, self.PhiD.shape[-1]))

        F = np.hstack((self.PhiHalf[..., 0], self.PhiTilde[..., 0] + self.dt/2 * self.GTilde[..., 0] ))
        pinvF = np.linalg.pinv(F, rcond=self.rcond)
        K = np.zeros((self.d, self.d, self.Nt-1))
        B = np.zeros((self.d, self.dTilde, self.Nt-1))

        for n in range(self.Nt-1):
            Z = self.PhiD[..., n] - self.PhiHalf[...,n] @ self.R[...,n].T
            if n == 0:
                X = pinvF @ Z
                K[..., n], B[..., n] = X[:self.d,:].T/self.dt*2, X[self.d:,:].T
            else:
                for k in range(1,n):
                    Z += -self.dt * self.PhiHalf[..., k] @ K[..., n-k].T
                    Z += -self.dt *  self.GTilde[..., k] @  B[..., n-k].T
                Z += -self.dt/2 * self.PhiHalf[..., n] @ K[..., 0].T
                Z += -self.dt * self.GTilde[..., n] @  B[..., 0].T 
                X = pinvF @ Z
                K[..., n], B[..., n] = X[:self.d,:].T/self.dt, X[self.d:,:].T

        return K, B


    def solve(self, Phi0, PhiTilde0, G, GTilde):
        sol = np.zeros((Phi0.shape[0], Phi0.shape[1], self.Nt))
        sol[..., 0] = Phi0
        for n in range(self.Nt-1):
            mat = np.eye(self.R.shape[0]) - self.dt/2*self.R[...,n].T - self.dt**2/4 * self.K[..., 0].T
            matp = np.eye(self.R.shape[0]) + self.dt/2 * self.R[...,n].T + self.dt**2/4 * self.K[..., 0].T
            rhs = sol[...,n] @ matp + self.dt * (PhiTilde0 + self.dt/2 * GTilde[..., 0]) @ self.B[...,n].T 
            rhs += self.dt * G[...,n]
            for k in range(n):
                 rhs += self.dt**2 * (sol[...,k]+sol[...,k+1])/2 @ self.K[...,n-k].T
                 rhs += self.dt**2 * GTilde[...,k] @ self.B[...,n-k].T
            rhs += -self.dt**2 * GTilde[..., 0] @ self.B[...,n].T
            rhs += self.dt**2 * GTilde[...,n] @ self.B[..., 0].T
            sol[..., n+1] = np.linalg.solve(mat.T, rhs.T).T
            
        return sol


class ROM_Partial_Data(abc.ABC):

    rcond = np.finfo(float).eps

    def __init__(self, Phi, PhiTilde, PhiD, dt):

        self.Phi, self.PhiTilde, self.PhiD = Phi, PhiTilde, PhiD
        self.dt = dt

        self.PhiHalf = 0.5 * (self.Phi[..., :-1] + self.Phi[..., 1:])
        self.PhiTildeHalf = 0.5 * (self.PhiTilde[..., :-1] + self.PhiTilde[..., 1:])

        self.Ns, self.d, self.Nt = Phi.shape
        self.dTilde = PhiTilde.shape[1]
        self.N = self.d + self.dTilde

        self.R      = NotImplemented
        self.RTilde = NotImplemented

        self.K = NotImplemented
        self.B = NotImplemented

    def solveConstantR(self):
        
        K = np.zeros((self.d, self.d, self.PhiD.shape[-1]))
        B = np.zeros((self.d, self.dTilde, self.PhiD.shape[-1]))

        F = np.hstack((self.PhiHalf[..., 0], self.PhiTilde[..., 0]))
        pinvF = np.linalg.pinv(F, rcond=self.rcond)

        for n in range(self.Nt-1):
            Z = self.PhiD[..., n].copy()
            if n == 0:
                X = pinvF @ Z
                # K[..., 0] = R + dt/2 K0
                
                K[..., 0], B[..., 0] = X[:self.d,:].T, X[self.d:,:].T
            else:
                for k in range(1,n+1):
                    Z += -self.PhiHalf[..., k] @ K[..., n-k].T
                X = pinvF @ Z
                K[..., n], B[..., n] = X[:self.d,:].T, X[self.d:,:].T
        # Needed for the correct factor in self.solve()
        K[..., 0] *= 2
        K *= 1/self.dt
                
        return K, B

    def solveRKB(self, regR, regK):
        R    = np.zeros((self.d, self.d, self.PhiD.shape[-1])) 
        K    = np.zeros((self.d, self.d, self.PhiD.shape[-1])) 
        B    = np.zeros((self.d, self.dTilde, self.PhiD.shape[-1])) 

        m01  = self.PhiHalf[..., 0]
        m12  = self.PhiHalf[...,1]

        Zp   = np.zeros_like(self.Phi[..., 0])
        Zt   = np.zeros_like(self.PhiTilde[..., 0])
        I    = np.eye(self.Phi.shape[1])
        Zdd  = np.zeros((self.d, self.d))
        Zddt = np.zeros((self.d, self.dTilde))

        PhiT0 = self.PhiTilde[..., 0]

        F1 = np.hstack((m01, (self.dt/2)*m01, PhiT0, Zp, Zp, Zt))
        F2 = np.hstack((Zp, (self.dt/2)*m12, Zt, m12, self.dt*m01, PhiT0))
        F3 = np.hstack((-regR*I, Zdd, Zddt,  regR*I, Zdd, Zddt))
        F4 = np.hstack(( Zdd, -regK*I, Zddt, Zdd,  regK*I, Zddt))

        F  = np.vstack((F1, F2, F3, F4))
        pinvF = np.linalg.pinv(F, rcond=self.rcond)

        Z  = np.vstack((self.PhiD[..., 0], 
                        self.PhiD[..., 1],
                        np.zeros_like(R[..., 0]),
                        np.zeros_like(R[..., 0])))
        
        X = pinvF @ Z

        R[..., 0] = X[:self.d,:].T
        K[..., 0] = X[self.d:2*self.d,:].T
        B[..., 0] = X[2*self.d:2*self.d+self.dTilde,:].T
        R[...,1] = X[2*self.d+self.dTilde:3*self.d+self.dTilde,:].T
        K[...,1] = X[3*self.d+self.dTilde:4*self.d+self.dTilde,:].T
        B[...,1] = X[4*self.d+self.dTilde:,:].T

        for n in range(2, self.Nt-1):
            F1 = np.hstack((
                self.PhiHalf[..., n], 
                self.dt*self.PhiHalf[..., 0], 
                self.PhiTilde[..., 0]
            ))
            F2 = np.hstack((regR*np.eye(self.d), Zdd, Zddt))
            F3 = np.hstack((Zdd, regK*np.eye(self.d), Zddt))
            F  = np.vstack((F1, F2, F3))
            
            pinvF = np.linalg.pinv(F, rcond=self.rcond)
            
            b = self.PhiD[..., n].copy()
            for k in range(1,n):
                # K contains a factor of dt
                b += -self.dt*self.PhiHalf[..., k] @K[..., n-k].T
            b += - self.dt/2 * self.PhiHalf[..., n]@K[..., 0].T
            Z = np.vstack((b, regR* R[..., n-1], regK* K[..., n-1]))
            
            X = pinvF @ Z
            R[..., n] = X[:self.d,:].T
            K[..., n] = X[self.d:2*self.d,:].T
            B[..., n] = X[2*self.d:,:].T

        return R, K, B
        
    def train(self, constantR=True, regR=None, regK=None):
        if constantR:
            if (regR != None or regK != None):
                print("Regularization is ignored for constant R.")
            self.R  = np.zeros((self.d, self.d, self.PhiD.shape[-1])) 
            self.K, self.B = self.solveConstantR()
        else:
           if regR==None or regK==None:
               raise Exception("Need to specify the regularization parameters regR and regK")
           self.R, self.K, self.B = self.solveRKB(regR, regK)


    def solve(self, Phi0, PhiTilde0, dt, Nt):
        

        sol = np.zeros((Phi0.shape[0], Phi0.shape[1], Nt+1))
        sol[..., 0] = Phi0
        
        for n in range(Nt):
            # K contains a factor of dt
            mat  = np.eye(self.K.shape[0])  - dt/2*self.R[..., n].T - dt**2/4 * self.K[..., 0].T
            matp = np.eye(self.K.shape[0])  + dt/2*self.R[..., n].T + dt**2/4 * self.K[..., 0].T
            matInv = np.linalg.inv(mat) 
            rhs = sol[..., n]@matp + dt*PhiTilde0@self.B[..., n].T 

            for k in range(n):
                 rhs += dt**2 * (sol[..., k]+sol[..., k+1])/2 @ self.K[..., n-k].T
            sol[..., n+1] = rhs @ matInv
        return sol
    
    def lstsq(A, b):
        X = np.linalg.pinv(A, rcond=self.rcond)@b
        return X
    

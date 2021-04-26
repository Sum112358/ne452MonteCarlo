import numpy as np

from typing import Dict, Tuple

def pot_func(phi: float, V:float) -> float:
    pot=V*(1.+np.cos(phi))
    return pot

def pot_funcS(phi: float, V: float) -> float:
    pot=V*(1.+np.sin(phi))
    return pot

def Vij(p1: float, p2:float, g:float) -> float:
    V12=g*(np.cos(p1-p2)-3.*np.cos(p1)*np.cos(p2))
    return V12

def Vijphi_rel(dphi:float, g:float) -> float:
    Vij=(-g/2.)*(np.cos(dphi))
    return Vij

def VijPhi_CM(dphi: float, g:float) -> float:
    Vij=(-g*3./2.)*(np.cos(dphi))
    return Vij

def pot_matrix(size: int) -> np.ndarray:
    V_mat=np.zeros((size,size),float)
    for m in range(size):
        for mp in range(size):
            if m==(mp+1) or m == (mp-1):
                V_mat[m,mp]=.5
    return V_mat

def gen_prob_dist(Ng: int, rho_phi: np.ndarray) -> np.ndarray:
    p = np.zeros((Ng, Ng, Ng),float)
##Normalize:
    P_norm = np.zeros((Ng,Ng),float)
    for i0 in range(Ng):
        for i1 in range(Ng):
            di01=i0 - i1
            if di01 < 0:
                di01+=Ng
            for i2 in range(Ng):
                di12= i1- i2
                if di12 < 0:
                    di12 +=Ng
                p[i0,i1,i2]=rho_phi[di01]*rho_phi[di12]
                P_norm[i0,i2] += p[i0,i1,i2]
    for i0 in range(Ng):
        for i1 in range(Ng):
            for i2 in range(Ng):
                p[i0,i1,i2]=p[i0,i1,i2]/P_norm[i0,i2]
    with open('pnorm.out', 'w') as f:
        f.write('Old Method\n')
        for i in range(Ng):
            for j in range(Ng):
                f.write(f"i={i},j={j}\t Mod Formula={((Ng-1)*j + i)% Ng}\t{P_norm[i,j]}\n")
    return p


def gen_prob_dist_end(Ng: int, rho_phi: np.ndarray) -> np.ndarray:
    p = np.zeros((Ng, Ng),float)
##Normalize:
    P_norm = np.zeros(Ng,float)
    for i0 in range(Ng):
        for i1 in range(Ng):
            di01=i0 - i1
            if di01 < 0:
                di01+=Ng
            p[i0,i1]=rho_phi[di01]
            P_norm[i0] += p[i0,i1]
    for i0 in range(Ng):
        for i1 in range(Ng):
            p[i0,i1]=p[i0,i1]/P_norm[i0]
    return p

class fastGenProbDist:
    """
    Supports cahcing of commonly used values. By default caching is disabled.
    Can be enabled by passing the flag during initialization
    """
    def __init__(self, rho_phi: np.ndarray, Ng: int, caching:bool = False) -> None:
        self.__rho_phi: np.ndarray = rho_phi
        self.__Ng: int = Ng
        self.__genPNormFull() # Compute all the P Norms for PFull
        self.__genPNormEnd() # compute and cache the P norms for PEnd
        self.__caching = caching
        if self.__caching:
            self.__pFullCaching: Dict[Tuple[int, int, int], float] = {}
            self.__pEndCaching: Dict[Tuple[int, int, int], float] = {}

    def __genPNormFull(self) -> None:
        self.__P_norm = np.zeros(self.__Ng, dtype=float)
        for i in range(self.__Ng):
            di01 = 0
            di12 = i
            for _ in range(self.__Ng):
                di01 = (di01 - 1) % self.__Ng
                di12 = (di12 + 1) % self.__Ng
                self.__P_norm[i] += self.__rho_phi[di01]*self.__rho_phi[di12]
        with open('pnorm.out', 'a') as f:
            f.write('\n New Method\n')
            for i in range(self.__Ng):
                f.write(f"{self.__P_norm[i]}\n")
    
    def pFull(self, i0:int, i1:int, i2:int) -> float:
        if self.__caching:
            if (i0,i1,i2) in self.__pFullCaching:
                return self.__pFullCaching[(i0,i1,i2)]
            else:
                val = self.__calcPFull(i0, i1, i2)
                self.__pFullCaching[(i0,i1,i2)] = val
                return val
        else:
            return self.__calcPFull(i0, i1, i2)

    def __calcPFull(self, i0:int, i1:int, i2:int) -> float:
        # Index into P_norm at ((Ng-1)*j + i)% Ng
        ind = ((self.__Ng-1)*i2 + i0)% self.__Ng
        di01 = (i0 - i1) % self.__Ng
        di12= (i1- i2) % self.__Ng
        return self.__rho_phi[di01]*self.__rho_phi[di12] / self.__P_norm[ind]

    def __genPNormEnd(self) -> None:
        self.__P_End_norm = np.zeros(self.__Ng, dtype=float)
        for i in range(self.__Ng):
            for j in range(self.__Ng):
                di01 = (i - j) % self.__Ng
                self.__P_End_norm[i] += self.__rho_phi[di01]

    def pEnd(self, i0:int, i1:int) -> float:
        if self.__caching:
            if (i0,i1) in self.__pEndCaching:
                return self.__pEndCaching[(i0,i1)]
            else:
                val = self.__calcPEnd(i0, i1)
                self.__pEndCaching[(i0,i1)] = val
                return val
        else:
            return self.__calcPEnd(i0,i1)

    def __calcPEnd(self, i0:int, i1:int):
        # Index into P_End_norm at i0.
        di01 = (i0 - i1) % self.__Ng
        return self.__rho_phi[di01] / self.__P_End_norm[i0]

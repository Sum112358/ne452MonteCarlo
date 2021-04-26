import json
from io import TextIOWrapper
from numpy import pi

class mcParameters:
    PIGS: bool = False 
    T : float = 1.0 #temperature in Kelvin
    beta: float = None # in K^-1
    m_max: int = 10
    Ngrid: int = None # Size of the grid
    P: int = 9 # number of beads
    tau: float = None 
    B: float = 1 #Rotational Constant in 1/cm
    V0: float = 1 #potential
    g: float = 1
    N_accept: int = 0
    N_total: int = 0
    MC_steps: int = 10000 # number of MC steps
    Nskip: int = 100 #skip steps for trajectory
    N: int = 2 #number of rotors
    delta_phi: float = None
    fileloc: str = 'files/'
    def __init__(self, file: TextIOWrapper) -> None:
        data = json.load(file)
        # sanatize the json
        for key in data:
            if not hasattr(self, key): # sanatize the input json
                raise Exception("Invalid JSON, key={key} is not a valid parameter")
            setattr(self, key, data[key])
        ### Parameter Setup
        self.beta = 1.0/self.T
        self.Ngrid = 2*self.m_max + 1
        self.tau = self.beta/self.P
        self.delta_phi = 2 * pi/float(self.Ngrid)

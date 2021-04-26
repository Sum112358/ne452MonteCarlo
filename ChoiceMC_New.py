import numpy as np
from numpy.random import default_rng
#import matplotlib.pyplot as plt
import MC_function as MCf
from mcParameters import mcParameters
import os
import time


start = time.time()
### Parameter Input ###
"""
Parameters are passed in as a Json Argument File. Doing this allows for changes in the arguments to be modified in the JSON and for JSON itrations to be run.
"""
with open('parameters.json', 'r') as f:
    p = mcParameters(f)

### Memory Allocation ###    
rho_phi=np.zeros(p.Ngrid,float)
rho_phi_pqc=np.zeros(p.Ngrid,float)
rho_mmp=np.zeros((p.Ngrid,p.Ngrid),float)

psi_m_phi=np.zeros((p.Ngrid,p.Ngrid),float)
psi_phi=np.zeros((p.Ngrid,p.Ngrid),float)

rhoV=np.zeros((p.Ngrid),float)
rhoVij=np.zeros((p.Ngrid,p.Ngrid),float)
rho_free=np.zeros((p.Ngrid,p.Ngrid),float)
rho_potential=np.zeros(p.Ngrid,float)
potential=np.zeros(p.Ngrid,float)
rho_tau=np.zeros((p.Ngrid,p.Ngrid),float)

histo_L=np.zeros(p.Ngrid,float)
histo_R=np.zeros(p.Ngrid,float)
histo_middle=np.zeros(p.Ngrid,float)
histo_pimc=np.zeros(p.Ngrid,float)

prob_p=np.zeros(p.Ngrid,float)

path_phi=np.zeros((p.N,p.P),int) ## i  N => number of beads
prob_full=np.zeros(p.Ngrid,float)

### Logging ###
if not os.path.exists(f'./{p.fileloc}'):
    os.makedirs('./files') # if a files folder does not exist create it
with open('log.out', 'w') as f:
    f.write("") # create the log.out file to handle all sys.out calls


### Solve the 1 Body Hamiltonian ###

# Build rho vs dphi (density matrix) grid
V=p.V0 * MCf.pot_matrix(p.Ngrid)
H=V.copy()

for m in range(p.Ngrid):
    m_value = -p.m_max+m
    H[m,m] = p.B*float(m_value**2)+p.V0 # constant potential term on diagonal
evals, evecs =np.linalg.eigh(H)

Z_exact=0.  #sum over state method
for m in range(p.Ngrid):
    Z_exact+=np.exp(-p.beta*evals[m])
    for mp in range(p.Ngrid):
        for n in range(p.Ngrid):
            rho_mmp[m,mp]+=np.exp(-p.beta*evals[n])*evecs[m,n]*evecs[mp,n]
         
Z_exact_pigs=rho_mmp[p.m_max,p.m_max]
rho_dot_V_mmp=np.dot(rho_mmp,H)
E0_pigs_sos=rho_dot_V_mmp[p.m_max,p.m_max]

with open('log.out', 'a') as f:
    f.write(f"Z (sos) = {Z_exact}\n")
    f.write(f"A (sos) = {-(1./p.beta)*np.log(Z_exact)}\n")
    f.write(f"E0 (sos) = {evals[0]}\n")
    f.write(f"E0 (pigs sos) = {E0_pigs_sos/Z_exact_pigs}\n")
    f.write("\n")

# <phi|m><m|n> exp(-beta E n) <n|m'><m'|phi>
with open(f'{p.fileloc}rho_sos', 'w') as rho_sos_out:
    #build basis
    for i in range(p.Ngrid):
        for m in range(p.Ngrid):
            m_value=-p.m_max+m
            psi_m_phi[i,m]=np.cos(i*p.delta_phi*m_value)/np.sqrt(2.*np.pi)
            
    for i in range(p.Ngrid):
        for n in range(p.Ngrid):
            for m in range(p.Ngrid):
                psi_phi[i,n]+=evecs[m,n]*psi_m_phi[i,m]

    for i in range(p.Ngrid):
        rho_exact=0.   
        for n in range(p.Ngrid):
            rho_exact+=np.exp(-p.beta*evals[n])*(psi_phi[i,n]**2)
        rho_exact/=(Z_exact)
        rho_sos_out.write(f"{i*p.delta_phi} {rho_exact} {psi_phi[i,0]**2}\n")

#free rotor density matrices below

with open(f'{p.fileloc}rhofree_sos', 'w') as rhofree_sos_out:
    for i in range(p.Ngrid):
        dphi=float(i)*p.delta_phi
        integral=0.
        for m in range(1,p.m_max):
            integral+=(2.*np.cos(float(m)*dphi))*np.exp(-p.tau*p.B*m**2)
        integral=integral/(2.*np.pi)
        integral=integral + 1./(2.*np.pi)
        rho_phi[i]=np.fabs(integral)
        rhofree_sos_out.write(str(dphi)+' '+str(rho_phi[i])+'\n')

with open(f'{p.fileloc}rhofree_pqc', 'w') as rho_pqc_out:
    # PQC rho
    for i in range(p.Ngrid):
        dphi=float(i)*p.delta_phi
        rho_phi_pqc[i]=np.sqrt(1./(4.*np.pi*p.B*p.tau))*np.exp(-1./(2.*p.tau*p.B)*(1.-np.cos(dphi)))
        rho_pqc_out.write(f"{dphi} {rho_phi_pqc[i]}\n")

# marx method; most accurate
    # marx muser
with open(f'{p.fileloc}rhofree_marx', 'w') as rho_marx_out:
    for i in range(p.Ngrid):
        dphi=float(i)*p.delta_phi
        integral=0.
        for m in range(p.m_max):
            integral+=np.exp(-1./(4.*p.tau*p.B)*(dphi+2.*np.pi*float(m))**2)
        for m in range(1,p.m_max):
            integral+=np.exp(-1./(4.*p.tau*p.B)*(dphi+2.*np.pi*float(-m))**2)
        integral*=np.sqrt(1./(4.*np.pi*p.B*p.tau))
        rho_phi[i]=integral
        rho_marx_out.write(f"{dphi} {integral}\n")

# potential rho
for i_new in range(p.Ngrid):
    rhoV[i_new] = np.exp(- p.tau*(MCf.pot_funcS(float(i_new)*p.delta_phi,p.V0)))

# rho pair
for i in range(p.Ngrid):
    for j in range(p.Ngrid):
        rhoVij[i,j]=np.exp(-p.tau*(MCf.Vij(i*p.delta_phi,j*p.delta_phi,p.g)))

# NMM results
for i in range(p.Ngrid):
    potential[i]=MCf.pot_func(float(i)*p.delta_phi,p.V0)
    rho_potential[i]=np.exp(-(p.tau/2.)*potential[i])
    for ip in range(p.Ngrid):
        integral=0.
        dphi=float(i-ip)*p.delta_phi
        for m in range(p.m_max):
            integral+=np.exp(-1./(4.*p.tau*p.B)*(dphi+2.*np.pi*float(m))**2)
        for m in range(1,p.m_max):
            integral+=np.exp(-1./(4.*p.tau*p.B)*(dphi+2.*np.pi*float(-m))**2)
        integral*=np.sqrt(1./(4.*np.pi*p.B*p.tau))
        rho_free[i,ip]=integral

#output potential to a file
with open(f'{p.fileloc}V', 'w') as potential_out:
    for i in range(p.Ngrid):
            potential_out.write(f"{(float(i)*p.delta_phi)} {(potential[i])}\n")

# construct the high temperature density matrix
for i1 in range(p.Ngrid):
        for i2 in range(p.Ngrid):
                rho_tau[i1,i2]=rho_potential[i1]*rho_free[i1,i2]*rho_potential[i2]

# form the density matrix via matrix multiplication
#set initial value of rho
rho_beta=rho_tau.copy()

for k in range(p.P-1):
    rho_beta=p.delta_phi*np.dot(rho_beta,rho_tau)

E0_nmm=0.
rho_dot_V=np.dot(rho_beta,potential)
Z0=0. # pigs pseudo Z

with open(f'{p.fileloc}rho_nmm', 'w') as rho_nmm_out:
    Z_nmm=rho_beta.trace()*p.delta_phi # thermal Z
    for i in range(p.Ngrid):
        E0_nmm+=rho_dot_V[i]
        for ip in range(p.Ngrid):
            Z0+=rho_beta[i,ip]
            rho_nmm_out.write(f"{i*p.delta_phi} {rho_beta[i,i]/Z_nmm}\n")
    E0_nmm/=Z0

with open(f'log.out', 'a') as f:
    f.write(f"Z (tau) = {Z_nmm}\n")
    f.write(f"Z (tau) = {E0_nmm}\n")
    f.write('\n')

with open(f'{p.fileloc}Evst', 'a') as E0_vs_tau_out:
    E0_vs_tau_out.write(f"{p.tau} {E0_nmm}\n")

#distribution = MCf.fastGenProbDist(rho_phi, p.Ngrid, True)
distribution = MCf.fastGenProbDist(rho_phi, p.Ngrid, False)

#p_dist=MCf.gen_prob_dist(p.Ngrid,rho_phi)
#p_dist_end=MCf.gen_prob_dist_end(p.Ngrid,rho_phi)

for i in range(p.N):
    for j in range(p.P): # set path at potential minimum
        path_phi[i,j]=int(p.Ngrid/2)
        path_phi[i,j]=0
        path_phi[i,j]=np.random.randint(p.Ngrid)

# recommanded numpy random number initialization
rng = default_rng()

print(f"Runtime After Setup Before MC = {time.time() - start}")

with open('log.out', 'a') as f:
    f.write("start MC\n")

with open(f'{p.fileloc}traj_A.dat', 'w') as traj_out: # open the file here so we don't pay the cost each iteration
    V_average=0.
    E_average=0.
    for n in range(p.MC_steps):
        V_total=0.
        E_total=0.
        for i in range(p.N):
            for j in range(p.P): 
                # p_minus=j-1
                # p_plus=j+1
                # if (p_minus<0):
                #     p_minus+=p.P
                # if (p_plus>=p.P):
                #     p_plus-=p.P  
                # p_minus=j-1
                # p_plus=j+1
                p_minus = (j-1) % p.P
                p_plus = (j+1) % p.P
                
                if p.PIGS:
                    # kinetic action
                    if j==0:
                        for ip in range(p.Ngrid):
                            prob_full[ip]=p_dist_end[ip,path_phi[i,p_plus]]
                            #prob_full[ip] = distribution.pEnd(ip,path_phi[i,p_plus])
                    if j==(p.P-1):
                        for ip in range(p.Ngrid):
                            prob_full[ip]=p_dist_end[path_phi[i,p_minus],ip]
                            #prob_full[ip] = distribution.pEnd(path_phi[i,p_minus],ip)
                    if (j!=0 and j!= (p.P-1)):
                        for ip in range(p.Ngrid):
                            prob_full[ip]=p_dist[path_phi[i,p_minus],ip,path_phi[i,p_plus]]
                            #prob_full[ip] = distribution.pFull(path_phi[i,p_minus],ip,path_phi[i,p_plus])
                else:
                    for ip in range(p.Ngrid):
                        #prob_full[ip]=p_dist[path_phi[i,p_minus],ip,path_phi[i,p_plus]]
                        prob_full[ip] = distribution.pFull(path_phi[i,p_minus],ip,path_phi[i,p_plus])
                        prob_full[ip]*=rhoV[ip] # local on site interaction

                # NN interactions and PBC(periodic boundary condistions)
                if (i<(p.N-1)):
                    for ir in range(len(prob_full)):
                        prob_full[ir]*=rhoVij[ir,path_phi[i+1,j]]
                if (i>0):
                    for ir in range(len(prob_full)):
                        prob_full[ir]*=rhoVij[ir,path_phi[i-1,j]]
                        # BPC below
                if (i==0):
                    for ir in range(len(prob_full)):
                        prob_full[ir]*=rhoVij[ir,path_phi[p.N-1,j]]
                if (i==(p.N-1)):
                    for ir in range(len(prob_full)):
                        prob_full[ir]*=rhoVij[ir,path_phi[0,j]]

                #normalize
                norm_pro=0.
                for ir in range(len(prob_full)):
                    norm_pro+=prob_full[ir]
                for ir in range(len(prob_full)):
                    prob_full[ir]/=norm_pro
                index=rng.choice(p.Ngrid,1, p=prob_full)

                # Note it is faster to use the Numpy distribution for this once your samples get > 10,000. See benchmarking here:
                #http://cgi.cs.mcgill.ca/~enewel3/posts/alias-method/index.html
                #on small distributions it is better to use the own implementation of an alias method

                path_phi[i,j] = index
                    
                p.N_total+=1

                histo_pimc[path_phi[i,j]]+=1.
            
            if (n % p.Nskip==0):
                traj_out.write(f"{path_phi[i,0]*p.delta_phi} ")
                traj_out.write(f"{path_phi[i,p.P-1]*p.delta_phi} ")
                traj_out.write(f"{path_phi[i,int((p.P-1)/2)]*p.delta_phi} ") #middle bead
                traj_out.write('\n')
                
            histo_L[path_phi[i,0]]+=1.
            histo_R[path_phi[i,p.P-1]]+=1.
            histo_middle[path_phi[i,int((p.P-1)/2)]]+=1.

            V_total+= MCf.pot_func(float(path_phi[i,int((p.P-1)/2)])*p.delta_phi,p.V0)
            E_total+= MCf.pot_func(float(path_phi[i,0])*p.delta_phi,p.V0)
            E_total+= MCf.pot_func(float(path_phi[i,p.P-1])*p.delta_phi,p.V0)

        V_average+=V_total
        E_average+=E_total

with open('log.out', 'a') as f:
    f.write(f"<V> = {str((V_average / p.MC_steps) / p.N)}\n")
    f.write(f"<E> = {str((E_average / p.MC_steps) / p.N)}\n")

with open(f"{p.fileloc}histo_A_P{p.P}_N{p.N}", 'w') as histo_out:
    for i in range(p.Ngrid):
        histo_out.write(f"{i*p.delta_phi} {histo_pimc[i]/(p.MC_steps*p.N*p.P)/p.delta_phi} {histo_middle[i]/(p.MC_steps*p.N)/p.delta_phi}\n")
        histo_out.write(f" {histo_L[i]/(p.MC_steps*p.N)/p.delta_phi}\n")
        histo_out.write(f"{histo_R[i]/(p.MC_steps*p.N)/p.delta_phi}\n")


import os, psutil 
process = psutil.Process(os.getpid())
print(f"Total Memory = {process.memory_info().rss}")
print(f"Runtime = {time.time() - start}")


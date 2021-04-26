import numpy as np
from numpy.random import default_rng
#import matplotlib.pyplot as plt
import MC_function as MCf
import time

start = time.time()


##
#PIGS=True
PIGS=False
T=1.    # temperature in Kelvin
beta=1./T   # in K^-1

m_max=100
Ngrid=2*m_max+1
P=9  # number of beads
tau=beta/float(P)
B=1.   # rotational constant in 1/CM (required for linden.x)
V0=1. # potential
g=1.
N_accept=0
N_total=0
MC_steps=10000
Nskip=100 # for trajectory
# number of rotors
N=2

rho_phi=np.zeros(Ngrid,float)
rho_phi_pqc=np.zeros(Ngrid,float)
## Build rho vs dphi (density matrix) grid
delta_phi=2.*np.pi/float(Ngrid)


# 1 body Hamiltonian
V=V0 * MCf.pot_matrix(2*m_max+1)
H=V.copy()

for m in range(2*m_max+1):
    m_value=-m_max+m
    H[m,m]=B*float(m_value**2)+V0 # constant potential term on diagonal
evals, evecs =np.linalg.eigh(H)

rho_mmp=np.zeros((2*m_max+1,2*m_max+1),float)

Z_exact=0.  #sum over state method
for m in range(2*m_max+1):
    Z_exact+=np.exp(-beta*evals[m])
    for mp in range(2*m_max+1):
        for n in range(2*m_max+1):
            rho_mmp[m,mp]+=np.exp(-beta*evals[n])*evecs[m,n]*evecs[mp,n]
         
Z_exact_pigs=rho_mmp[m_max,m_max]

rho_dot_V_mmp=np.dot(rho_mmp,H)
E0_pigs_sos=rho_dot_V_mmp[m_max,m_max]


print('Z (sos) = ',Z_exact)
print('A (sos) = ',-(1./beta)*np.log(Z_exact))
print('E0 (sos) =',evals[0])
print('E0 (pigs sos) =',E0_pigs_sos/Z_exact_pigs)

print(' ')

# <phi|m><m|n> exp(-beta E n) <n|m'><m'|phi>
rho_sos_out=open('rho_sos','w')

#built basis
psi_m_phi=np.zeros((Ngrid,2*m_max+1),float)
for i in range(Ngrid):
    for m in range(2*m_max+1):
        m_value=-m_max+m
        psi_m_phi[i,m]=np.cos(i*delta_phi*m_value)/np.sqrt(2.*np.pi)
        
psi_phi=np.zeros((Ngrid,2*m_max+1),float)
for i in range(Ngrid):
    for n in range(2*m_max+1):
        for m in range(2*m_max+1):
            psi_phi[i,n]+=evecs[m,n]*psi_m_phi[i,m]

for i in range(Ngrid):
    rho_exact=0.   
    for n in range(2*m_max+1):
        rho_exact+=np.exp(-beta*evals[n])*(psi_phi[i,n]**2)
    rho_exact/=(Z_exact)
    rho_sos_out.write(str(i*delta_phi)+ ' '+str(rho_exact)+' '+str(psi_phi[i,0]**2)+'\n')
rho_sos_out.close()

# free rotor density matrices below

rhofree_sos_out=open('rhofree_sos','w')
rho_pqc_out=open('rhofree_pqc','w')
for i in range(Ngrid):
    dphi=float(i)*delta_phi
    integral=0.
    for m in range(1,m_max):
        integral+=(2.*np.cos(float(m)*dphi))*np.exp(-tau*B*m**2)
    integral=integral/(2.*np.pi)
    integral=integral + 1./(2.*np.pi)
    rho_phi[i]=np.fabs(integral)
    rhofree_sos_out.write(str(dphi)+' '+str(rho_phi[i])+'\n')
rhofree_sos_out.close()
# PQC rho
for i in range(Ngrid):
    dphi=float(i)*delta_phi
    rho_phi_pqc[i]=np.sqrt(1./(4.*np.pi*B*tau))*np.exp(-1./(2.*tau*B)*(1.-np.cos(dphi)))
    rho_pqc_out.write(str(dphi)+' '+str(rho_phi_pqc[i])+'\n')
rho_pqc_out.close()

# marx method; most accurate
    # marx muser
rho_marx_out=open('rhofree_marx','w')
for i in range(Ngrid):
    dphi=float(i)*delta_phi
    integral=0.
    for m in range(m_max):
        integral+=np.exp(-1./(4.*tau*B)*(dphi+2.*np.pi*float(m))**2)
    for m in range(1,m_max):
        integral+=np.exp(-1./(4.*tau*B)*(dphi+2.*np.pi*float(-m))**2)
    integral*=np.sqrt(1./(4.*np.pi*B*tau))
    rho_phi[i]=integral
    rho_marx_out.write(str(dphi)+' '+str(integral)+'\n')
rho_marx_out.close()

# potential rho
rhoV=np.zeros((Ngrid),float)
for i_new in range(Ngrid):
    rhoV[i_new] = np.exp(- tau*(MCf.pot_funcS(float(i_new)*delta_phi,V0)))
# rho pair
rhoVij=np.zeros((Ngrid,Ngrid),float)
for i in range(Ngrid):
    for j in range(Ngrid):
        rhoVij[i,j]=np.exp(-tau*(MCf.Vij(i*delta_phi,j*delta_phi,g)))

# NMM results
rho_free=np.zeros((Ngrid,Ngrid),float)
rho_potential=np.zeros(Ngrid,float)
potential=np.zeros(Ngrid,float)
for i in range(Ngrid):
    potential[i]=MCf.pot_func(float(i)*delta_phi,V0)
    rho_potential[i]=np.exp(-(tau/2.)*potential[i])
    for ip in range(Ngrid):
        integral=0.
        dphi=float(i-ip)*delta_phi
        for m in range(m_max):
            integral+=np.exp(-1./(4.*tau*B)*(dphi+2.*np.pi*float(m))**2)
        for m in range(1,m_max):
            integral+=np.exp(-1./(4.*tau*B)*(dphi+2.*np.pi*float(-m))**2)
        integral*=np.sqrt(1./(4.*np.pi*B*tau))
        rho_free[i,ip]=integral
#output potential to a file
potential_out=open('V','w')
for i in range(Ngrid):
        potential_out.write(str(float(i)*delta_phi)+' '+str(potential[i])+'\n')
potential_out.close()
# construct the high temperature density matrix
rho_tau=np.zeros((Ngrid,Ngrid),float)
for i1 in range(Ngrid):
        for i2 in range(Ngrid):
                rho_tau[i1,i2]=rho_potential[i1]*rho_free[i1,i2]*rho_potential[i2]

# form the density matrix via matrix multiplication
#set initial value of rho
#rho_beta=np.zeros((size,size),float)

rho_beta=rho_tau.copy()

for k in range(P-1):
        rho_beta=delta_phi*np.dot(rho_beta,rho_tau)

E0_nmm=0.
rho_dot_V=np.dot(rho_beta,potential)
Z0=0. # pigs pseudo Z
rho_nmm_out=open('rho_nmm','w')
Z_nmm=rho_beta.trace()*delta_phi # thermal Z

for i in range(Ngrid):
    E0_nmm+=rho_dot_V[i]
    for ip in range(Ngrid):
        Z0+=rho_beta[i,ip]
        rho_nmm_out.write(str(i*delta_phi)+ ' '+str(rho_beta[i,i]/Z_nmm)+'\n')
rho_nmm_out.close()
E0_nmm/=Z0

print('Z (tau) = ',Z_nmm)
print('E0 (tau) = ',E0_nmm)

E0_vs_tau_out=open('Evst','a')
E0_vs_tau_out.write(str(tau)+' '+str(E0_nmm)+'\n')
E0_vs_tau_out.close()
print(' ')

histo_L=np.zeros(Ngrid,float)
histo_R=np.zeros(Ngrid,float)
histo_middle=np.zeros(Ngrid,float)

histo_pimc=np.zeros(Ngrid,float)

#prob_p=np.zeros(Ngrid,float)
p_test=np.zeros(P,float)

p_dist=MCf.gen_prob_dist(Ngrid,rho_phi)
p_dist_end=MCf.gen_prob_dist_end(Ngrid,rho_phi)


path_phi=np.zeros((N,P),int) ## i  N => number of beads
for i in range(N):
    for p in range(P): # set path at potential minimum
    #   path_phi[p]=int(Ngrid/2)
        path_phi[i,p]=int(Ngrid/2)
        path_phi[i,p]=0
        path_phi[i,p]=np.random.randint(Ngrid)

traj_out=open('traj_A.dat','w')

# recommanded numpy random number initialization
rng = default_rng()

print(f"Runtime After Setup Before MC = {time.time() - start}")

print('start MC')

V_average=0.
E_average=0.
prob_full=np.zeros(Ngrid,float)
for n in range(MC_steps):
    V_total=0.
    E_total=0.
    for i in range(N):
        for p in range(P): 
            p_minus=p-1
            p_plus=p+1
            if (p_minus<0):
                p_minus+=P
            if (p_plus>=P):
                p_plus-=P  
            
            if PIGS==True:
                # kinetic action
                if p==0:
                    for ip in range(Ngrid):
                        prob_full[ip]=p_dist_end[ip,path_phi[i,p_plus]]
                if p==(P-1):
                    for ip in range(Ngrid):
                        prob_full[ip]=p_dist_end[path_phi[i,p_minus],ip]
                if (p!=0 and p!= (P-1)):
                    for ip in range(Ngrid):
                        prob_full[ip]=p_dist[path_phi[i,p_minus],ip,path_phi[i,p_plus]]
            else:
                for ip in range(Ngrid):
                    prob_full[ip]=p_dist[path_phi[i,p_minus],ip,path_phi[i,p_plus]]
                    prob_full[ip]*=rhoV[ip] # local on site interaction

            # NN interactions and PBC(periodic boundary condistions)
            if (i<(N-1)):
                for ir in range(len(prob_full)):
                    prob_full[ir]*=rhoVij[ir,path_phi[i+1,p]]
            if (i>0):
                for ir in range(len(prob_full)):
                    prob_full[ir]*=rhoVij[ir,path_phi[i-1,p]]
                    # BPC below
            if (i==0):
                for ir in range(len(prob_full)):
                    prob_full[ir]*=rhoVij[ir,path_phi[N-1,p]]
            if (i==(N-1)):
                for ir in range(len(prob_full)):
                    prob_full[ir]*=rhoVij[ir,path_phi[0,p]]

            #normalize
            norm_pro=0.
            for ir in range(len(prob_full)):
                norm_pro+=prob_full[ir]
            for ir in range(len(prob_full)):
                prob_full[ir]/=norm_pro
            index=rng.choice(Ngrid,1, p=prob_full)

            path_phi[i,p] = index
                
            N_total+=1

            histo_pimc[path_phi[i,p]]+=1.
        
        if (n%Nskip==0):
            traj_out.write(str(path_phi[i,0]*delta_phi)+' ')
            traj_out.write(str(path_phi[i,P-1]*delta_phi)+' ')
            traj_out.write(str(path_phi[i,int((P-1)/2)]*delta_phi)+' ') #middle bead
            traj_out.write('\n')
            
        histo_L[path_phi[i,0]]+=1.
        histo_R[path_phi[i,P-1]]+=1.
        histo_middle[path_phi[i,int((P-1)/2)]]+=1.

        V_total+= MCf.pot_func(float(path_phi[i,int((P-1)/2)])*delta_phi,V0)
        E_total+= MCf.pot_func(float(path_phi[i,0])*delta_phi,V0)
        E_total+= MCf.pot_func(float(path_phi[i,P-1])*delta_phi,V0)

    V_average+=V_total
    E_average+=E_total

traj_out.close()

print('<V> = ',V_average/MC_steps/N)
print('<E> = ',E_average/MC_steps/2./N)
histo_out=open('histo_A_P'+str(P)+'_N'+str(N),'w')
for i in range(Ngrid):
  histo_out.write(str(i*delta_phi)+' '+str(histo_pimc[i]/(MC_steps*N*P)/delta_phi) +' '+str(histo_middle[i]/(MC_steps*N)/delta_phi))
  histo_out.write(' '+str(histo_L[i]/(MC_steps*N)/delta_phi))
  histo_out.write(' '+str(histo_R[i]/(MC_steps*N)/delta_phi)+'\n')
histo_out.close()
#print('N_accept/N_total',N_accept/N_total)

import os, psutil 
process = psutil.Process(os.getpid())
print(f"Total Memory = {process.memory_info().rss}")
print(f"Runtime = {time.time() - start}")
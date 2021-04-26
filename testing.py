from numpy import testing
import MC_function as MCf
import numpy as np

# size = 6

# testing_rho_phi = np.zeros(21, dtype=float)

# for i in range(len(testing_rho_phi)):
#     testing_rho_phi[i] = np.random.rand()



# p_dist_old=MCf.gen_prob_dist(size,testing_rho_phi)


# newP_dist = MCf.fastGenProbDist(testing_rho_phi, size)


# for i in range(size):
#     for j in range(size):
#         for k in range(size):
#             if np.round(p_dist_old[i,j,k], 4) != np.round(newP_dist.pFull(i,j,k), 4):
#                 raise Exception("Bad, mistake")

# Ng = 3
# old = []
# for i in range(Ng):
#     for j in range(Ng):
#         di01 = i -j
#         #di01 = i - j % Ng
#         if di01 < 0:
#            di01 +=Ng
#         old.append(di01)

# new = []
# for i in range(Ng):
#     for j in range(Ng):
#         di01 = (i - j) % Ng
#         new.append(di01)

# for i in range(len(old)):
#     print(f"Old={old[i]}\t New={new[i]}")

k = 4
for p in range(k):
    p_minus = p-1
    p_plus = p+1
    if (p_minus<0):
        p_minus+=k
    if (p_plus>=k):
        p_plus-=k  
    print(f"p_minus={p_minus}\t p_plus={p_plus}")

print("new P formula")
for p in range(k):
    p_minus = (p-1) % k
    p_plus =  (p + 1) % k
    print(f"p_minus={p_minus}\t p_plus={p_plus}")
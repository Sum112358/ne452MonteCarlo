m_max = [10,20,50,70,100]
runtimeFull_origin = [17.9998, 27.9861, 66.5362019, 94.045255, 163.0765540599823]
runtimeFull_noCache = [23.7873, 42.71768, 103.33157, 138.172, 225.94329285621643]
runtimeFull_Cache = [19.88372, 30.4574, 71.97495, 98.36411, 163.1879847049713]

runTimeStart_origin = [0.116971, 0.51734, 5.786656, 16.15904, 46.130391]
runTimeStart_noCache = [0.076119, 0.4319, 4.78064, 12.38198, 33.426738]
runTimeStart_Cache = [0.07104, 0.4319, 5.055654, 11.4259, 34.8849]

Memory_origin = [26222592, 27017216, 35958784, 51326976, 96423936]
Memory_noCache = [26378240, 26619904, 27848704, 28889088, 31354880]
Memory_Cache = [27693056, 35876864, 133394432, 323649536, 754139136]

runtimeCpp = [1.68, 3.35, 7.40, 11.82, 19.93]
memoryCpp = [2328*1000, 3032*1000, 12128*1000, 27608*1000, 72616*1000]

from matplotlib import pyplot as plt

plt.plot(m_max, runtimeFull_origin, label='Original')
plt.plot(m_max, runtimeCpp , label='C++')
plt.xlabel("m_max")
plt.ylabel("Time (s)")
plt.title("Runtime C++ vs Original", y=1.0)
plt.legend()
plt.savefig('runTime_cpp.png')
plt.clf()

plt.plot(m_max, Memory_origin, label='Original')
plt.plot(m_max, memoryCpp , label='C++')
plt.xlabel("m_max")
plt.ylabel("Memory (bytes)")
plt.title("Memory C++ vs Original", y=1.0)
plt.legend()
plt.savefig('memory_cpp.png')
plt.clf()


# plt.plot(m_max, runtimeFull_origin, label='Original')
# plt.plot(m_max, runtimeFull_noCache, label='New no Caching')
# plt.plot(m_max, runtimeFull_Cache, label='New Caching')
# plt.xlabel("m_max")
# plt.ylabel("Time (s)")
# plt.title("Runtime of Simulation for Original, \nNew with Caching and New without Caching", y=1.0)
# plt.legend()
# plt.savefig('runtimeFull.png')
# plt.clf()

# plt.plot(m_max, runTimeStart_origin, label='Original')
# plt.plot(m_max, runTimeStart_noCache, label='New no Caching')
# plt.plot(m_max, runTimeStart_Cache, label='New Caching')
# plt.xlabel("m_max")
# plt.ylabel("Time (s)")
# plt.title("Runtime of Just Simulation Setup for Original, \nNew with Caching and New without Caching", y=1.0)
# plt.legend()
# plt.savefig('runtimeStart.png')
# plt.clf()

# plt.plot(m_max, Memory_origin, label='Original')
# plt.plot(m_max, Memory_noCache, label='New no Caching')
# plt.plot(m_max, Memory_Cache, label='New Caching')
# plt.xlabel("m_max")
# plt.ylabel("Memory Usage (bytes)")
# plt.title("Memory Usage for Original, New with Caching and New without Caching",y=1.08)
# plt.legend()
# plt.savefig('memory.png')
# plt.clf()
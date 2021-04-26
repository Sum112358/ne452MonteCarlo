from random import random
import numpy as np
import time
#from matplotlib import pyplot

from alias import aliasMethod

def makeDistribution(k: int):
    scores = [random() for _ in range(k)]
    total = sum(scores)
    probabilities = [s / total for s in scores]
    return probabilities

#make some random distribution and normalize it

def timeChoice(k) -> float:
    start = time.time()
    prob = makeDistribution(k)
    np.random.choice(k, p=prob)
    return time.time() -start

def timeChoice_draw(k, draw) -> float:
    start = time.time()
    prob = makeDistribution(k)
    for _ in range(draw):
        np.random.choice(k, p=prob)
    return time.time() -start

def timeAlias(k) -> float:
    start = time.time()
    prob = makeDistribution(k)
    aliasMethod(prob).draw()
    return time.time() - start

def timeAlias_draw(k, draw) -> float:
    start = time.time()
    prob = makeDistribution(k)
    for _ in range(draw):
        aliasMethod(prob).draw()
    return time.time() - start

from matplotlib import pyplot as plt
# Plot as the number of iterations increase for 1 draw
x = []
choice = []
alias = []
for k in range(1,6):
    x.append(10**k)
    choice.append(timeChoice(10**k))
    alias.append(timeAlias(10**k))

plt.plot(x, choice, label='Np.Choice')
plt.plot(x, alias, label='AliasMethod')
plt.title("Increasing Number of Iterations Drawing 1 From the Sample")
plt.xlabel("Number of Iterations")
plt.ylabel("Time (s)")
plt.legend()
plt.savefig("iterations.png")
plt.clf()

# Plot as the number of draws for i iteration
x_draw = []
choice_draw = []
alias_draw = []
for k in range(1000):
    x_draw.append(k)
    choice_draw.append(timeChoice_draw(10**3, k))
    alias_draw.append(timeAlias_draw(10**3, k))

plt.plot(x_draw, choice_draw, label='Np.Choice')
plt.plot(x_draw, alias_draw, label='AliasMethod')
plt.title("Increasing Number of Draws from 1 Iteration of the sample Size 10**3")
plt.xlabel("Number of Draws from the Sample")
plt.ylabel("Time (s)")
plt.legend()
plt.savefig("draws.png")
plt.clf()


#plot


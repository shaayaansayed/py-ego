from random import random
from math import pow
import numpy as np

pop = []
popFit = []

popSize = 20
generations = 50
tourSize = 5
encoding = 20

pRep = 0.2
pCr = 0.6
pMut = 0.4
pMutDash = 0.1 

nBits = int(pMutDash*encoding/pMut)

# user defined 
fitfunc = [] 
nvars = []
LB = []
UB = []

def ga(_fitfunc, _nvars, _LB, _UB):
    global fitfunc, nvars, LB, UB
    global pop, popFit, nBits

    fitfunc = _fitfunc
    nvars = _nvars 
    LB = _LB
    UB = _UB

    population()
    
    # main loop
    for i in range(0, generations):
        newPop = []
        newFit = []
        
        # save best individual in population
        bestFit = min(popFit)
        newPop.append(pop[popFit.index(bestFit)])
        newFit.append(bestFit)
        
        for k in range(1, popSize):
            # stochastically determine operation 
            op = multiflip(pRep, pCr)
            # tournament selection
            if op == 0:
                ind = tournament(popFit, tourSize)
                newPop.append(pop[ind])
                newFit.append(popFit[ind])
            # reproduction
            elif op == 1:
                parent1 = tournament(popFit, tourSize)
                parent2 = tournament(popFit, tourSize)
                offspring = crossover(pop[parent1], pop[parent2])
                offfit1 = fitness(decode(offspring[0]))
                offfit2 = fitness(decode(offspring[1]))
                newPop.append(offspring[0])
                newFit.append(offfit1)
                newPop.append(offspring[1])
                newFit.append(offfit2)
            # mutation
            else:
                indm = tournament(popFit, tourSize)
                newInd = mutate(pop[indm], nBits)
                fit = fitness(decode(newInd))
                newPop.append(newInd)
                newFit.append(fit)
        
        while len(pop) != len(newPop):
            worst = max(newFit)
            newPop.remove(newPop[newFit.index(worst)])   
            newFit.remove(worst)
        
        pop = newPop
        popFit = newFit
        
    return np.array(decode(pop[popFit.index(min(popFit))])), min(popFit)

def population():
    global pop, popFit

    pop = []
    popFit = []
    for i in xrange(0, popSize):
        ind = randInd()
        fit = fitness(decode(ind))

        pop.append(ind)
        popFit.append(fitness(decode(ind)))
                
def tournament(fit, size):
    global pop, popFit
    pool = []
    poolFit = []
    
    for i in range(0, size):
        rand = int(random() * popSize)
        pool.append(pop[rand])
        poolFit.append(popFit[rand])

    return poolFit.index(min(poolFit))  

def crossover(ind1, ind2):
    point = int(random() * encoding*nvars)
    off1 = ind1[0:point] + ind2[point:]
    off2 = ind2[0:point] + ind1[point:]
    return [off1, off2]      
 

def mutate(ind, nBits):   
    mIndex = int(random() * encoding*nvars)
    if ind[mIndex] == '0':
        newInd = ind[0:mIndex - 1] + '1' + ind[mIndex:]
    else:
        newInd = ind[0:mIndex - 1] + '0' + ind[mIndex:]

    newFit = fitness(decode(newInd))
    return newInd
         
def decode(genotype):
    phenotype = []
    for i in range(0, nvars):
        decode = int(genotype[encoding*i:encoding*(i + 1)], 2)
        gene = LB[i] + (decode)*(UB[i] - LB[i])/(pow(2, encoding) - 1)
        phenotype.append(gene)    
            
    return phenotype
    
def fitness(phenotype):
    f = fitfunc(np.array(phenotype))[0]
    return f

def randInd():
    global nvars

    ind = []
    for i in range(0, encoding*nvars):
        if random() < .5:
            ind.append('0')
        else:
            ind.append('1')
            
    return ''.join(ind)

def multiflip(pRep, pCr):
    rand = random()
    
    if rand < pRep:
        op = 0
    elif rand >= pRep and rand < pRep + pCr:
        op = 1
    else:
        op = 2
        
    return op
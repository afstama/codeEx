import numpy as np
import sys, getopt

def processData(dataset):
    with open(dataset) as f:
        lines = f.readlines()
    features = lines[0].strip().split(',')[:-1]
    cases = [lines[i].strip().split(',') for i in range(1, len(lines))]
    for j in range(len(cases)):
        cases[j] = [float(cases[j][i]) for i in range(len(cases[j]))]
    return features, cases

class Layer:
    def __init__(self, nNodes, nWeights, weights=None):
        if weights == None:
            self.W = [np.random.normal(0, 0.01, nWeights) for _ in range(nNodes)]
        else:
            self.W = weights
    def __repr__(self):
        return str(self.W)
    def __str__(self):
        return str(self.W)
    
class NN:
    def __init__(self, nodes, layers=None):
        # ex. nodes = [1,5,1]
        if layers == None:
            self.layers = []
            for i in range(1, len(nodes)):
                self.layers.append(Layer(nodes[i], nodes[i-1]+1))
        else:
            self.layers = layers

    def evaluate(self, case):
        X = (case.copy())[:-1]
        for i in range(len(self.layers)):
            layer: Layer = self.layers[i]
            X.append(1)
            Y = np.matmul(layer.W, X)
            X = list(Y)
            if i != len(self.layers)-1:
                X = [1/(1+np.e**(-X[j])) for j in range(len(X))]
        # print(X)
        return (case[-1] - X[0])**2
    
class GA:
    def __init__(self, trainD, testD, nn, popsize, elitism, p, K, iter):
        self.popsize = popsize
        self.elitism = elitism
        self.p = p
        self.K = K
        self.iter = iter

        self.nn = nn.copy()
        features, self.cases = processData(trainD)
        self.nn.insert(0, len(features)); self.nn.append(1)

        self.population = [NN(self.nn) for _ in range(self.popsize)]
        self.train()
        self.test(testD)
        return

    def cross_mutate(self):
        idx = [i for i in range(self.popsize)]
        idx1 = np.random.choice(idx); idx.remove(idx1); idx2 = np.random.choice(idx)
        nn1: NN = self.population[idx1]; nn2: NN = self.population[idx2]

        newLayers = []
        for i in range(len(nn1.layers)):
            newLayer = []
            nn1Layer: Layer = nn1.layers[i]; nn2Layer: Layer = nn2.layers[i]
            for j in range(len(nn1Layer.W)):
                newLayer.append((nn1Layer.W[j]+nn2Layer.W[j])/2)
                newLayer[-1] += np.random.choice([0, 1], p=[self.p, 1-self.p]) * np.random.normal(0, self.K, len(newLayer[-1]))
            newLayers.append(Layer(0, 0, newLayer))
        return NN(self.nn, newLayers)
    
    def train(self):
        for it in range(self.iter):
            # t = time.time()
            errs = []
            for pop in self.population:
                err = 0
                for case in self.cases:
                    err += pop.evaluate(case)
                errs.append(err/len(self.cases))
            
            newPopulation = [self.cross_mutate() for _ in range(self.popsize-self.elitism)]
            for _ in range(self.elitism):
                bestErr = min(errs)
                newPopulation.append(self.population[errs.index(bestErr)])
                self.population.remove(newPopulation[-1])
                errs.remove(bestErr)
            self.population = newPopulation.copy()

            # print(it)
            if (it+1) % 2000 == 0:
                print(f'[Train error @{it+1}]: {min(errs):.6f}')
            # print(time.time()-t)
    
    def test(self, dataset):
        cases = processData(dataset)[1]
        errs = []
        for pop in self.population:
            err = 0
            for case in cases:
                err += pop.evaluate(case)
            errs.append(err/len(cases))
        print(f'[Test error]: {min(errs):.6f}')
            

argv = sys.argv[1:]
opts, args = getopt.getopt(argv, '', ['train=', 'test=', 'nn=', 'popsize=', 'elitism=', 'p=', 'K=', 'iter='])
for opt, arg in opts:
    if opt == '--train':
        trainD = arg
    if opt == '--test':
        testD = arg
    if opt == '--nn':
        nnStr = arg.split('s')
        nn = []
        for ns in nnStr:
            if ns != '':
                nn.append(int(ns))
    if opt == '--popsize':
        popsize = int(arg)
    if opt == '--elitism':
        elitism = int(arg)
    if opt == '--p':
        p = float(arg)
    if opt == '--K':
        K = float(arg)
    if opt == '--iter':
        iter = int(arg)

ga = GA(trainD, testD, nn, popsize, elitism, p, K, iter)


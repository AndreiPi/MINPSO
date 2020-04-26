import functools
import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection
from tensorflow.keras.utils import to_categorical

import pso
import ann
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt

def dim_weights(shape):
    dim = 0
    for i in range(len(shape)-1):
        dim = dim + (shape[i] + 1) * shape[i+1]
    return dim

def vector_to_weights(vector, shape):
    weights = []
    idx = 0
    for i in range(len(shape)-1):
        r = shape[i+1]
        c = shape[i] + 1
        idx_min = idx
        idx_max = idx + r*c
        W = vector[idx_min:idx_max].reshape(r,c)
        weights.append(W)
    return weights

def eval_neural_network(weights, shape, X, y):
    mse = np.asarray([])
    for w in weights:
        weights = vector_to_weights(w, shape)
        nn = ann.MultiLayerPerceptron(shape, weights=weights)
        y_pred = nn.run(X)
        mse = np.append(mse, sklearn.metrics.mean_squared_error(np.atleast_2d(y), y_pred))
    return mse

def print_best_particle(best_particle):
    print("New best particle found at iteration #{i} with mean squared error: {score}".format(i=best_particle[0], score=best_particle[1]))


num_classes = 2
X_train, y_train, X_test, y_test = pkl.load(open("train_data4096.pkl", "rb"))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_inputs = X_train.shape[1]

y_true = np.array(y_train)

y_test_true = np.array(y_test)

# Set up
shape = (num_inputs, 64,32, num_classes)

cost_func = functools.partial(eval_neural_network, shape=shape, X=X_train, y=y_true.T)

swarm = pso.ParticleSwarm(cost_func, num_dimensions=dim_weights(shape), num_particles=50,decay=0.0013)

# Train...
i = 0
best_scores = [(i, swarm.best_score)]
print_best_particle(best_scores[-1])
f=open("ANNlosslog.txt","a")
scores=[]
while swarm.best_score>1e-6 and i<750:
    swarm._update()
    i = i+1
    scores.append(swarm.best_score)
    if swarm.best_score < best_scores[-1][1]:
        best_scores.append((i, swarm.best_score))
        print_best_particle(best_scores[-1])
        f.write(str(best_scores[-1][1])+" ")
f.flush()
f.write("\n------------\n")

fig, ax = plt.subplots()
ax.plot(range(len(scores)),scores )

ax.set(xlabel='Iteration)', ylabel='Loss ',
       title='ANN PSO loss plot')
ax.grid()
#ax.set_yscale('log')
ax.set_yticks(np.arange(0, 0.5, step=0.02))

fig.savefig("annpsolog1.png")



# Test...
best_weights = vector_to_weights(swarm.g, shape)
best_nn = ann.MultiLayerPerceptron(shape, weights=best_weights)
y_test_pred = np.round(best_nn.run(X_test))
print(sklearn.metrics.classification_report(y_test_true, y_test_pred.T))
f.write(sklearn.metrics.classification_report(y_test_true, y_test_pred.T))

plt.show()
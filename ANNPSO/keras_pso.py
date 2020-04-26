import functools
import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection
import pickle as pkl
from tensorflow.keras.utils import to_categorical
import pso
import ann
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib
import matplotlib.pyplot as plt

def build_model(loss):
    model = Sequential()
    model.add(Dense(25, activation='sigmoid', input_dim=4096, use_bias=False))
    model.add(Dense(50, activation='sigmoid', use_bias=False))
    model.add(Dense(2, activation='softmax', use_bias=False))

    model.compile(loss=loss,
                  optimizer='adam',metrics=['accuracy'])

    return model

LOSS = 'mse' # Loss function
BATCH_SIZE = 64 # Size of batches to train on

def vanilla_backpropagation(x_train, y_train):
    best_model = None
    best_score = 100.0

    for i in range(1):
        print(i)
        model_s = build_model(LOSS)
        model_s.fit(x_train, y_train,
                    epochs=5,
                    batch_size=BATCH_SIZE,verbose=0)
        train_score = model_s.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=0)
        if train_score[0] < best_score:
            best_model = model_s
            best_score = train_score[0]
    return best_model

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
        c = shape[i]
        idx_min = idx
        idx_max = idx + r*c
        W = vector[idx_min:idx_max].reshape(c,r)
        weights.append(W)
        #print(W.shape)
    return weights

def eval_neural_network(weights, shape, X, y, model,batchsize=BATCH_SIZE):
    mse = np.asarray([])
    for w in weights:
        weights = vector_to_weights(w, shape)
        #print(np.array(weights).shape)
        model.set_weights(weights)
        y_pred = model.evaluate(X,y,batch_size=batchsize ,verbose=0)
        mse = np.append(mse, y_pred[0])
    return mse

def print_best_particle(best_particle):
    print("New best particle found at iteration #{i} with mean squared error: {score}".format(i=best_particle[0], score=best_particle[1]))



X_train, y_train, X_test, y_test = pkl.load(open("train_data4096.pkl", "rb"))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_inputs = X_train.shape[1]
print(len(y_train))
y_true = np.array(y_train)
y_test_true = np.array(y_test)

#model_s = vanilla_backpropagation(X_train,y_train)
model_s = build_model(LOSS)
shapes =[]
# for i, layer in enumerate(model_s.get_weights()):
#     #shapes.append(layer.shape[1])
#     print(layer.shape)
# Set up
num_classes=2
shape = (num_inputs,25, 50, num_classes)
print(shape)
cost_func = functools.partial(eval_neural_network, shape=shape, X=X_train, y=y_train, model=model_s)

swarm = pso.ParticleSwarm(cost_func, num_dimensions=dim_weights(shape), num_particles=50)

# Train...
i = 0
best_scores = [(i, swarm.best_score)]
print_best_particle(best_scores[-1])
f=open("losskeras.txt","a")
scores = []
while swarm.best_score>1e-6 and i<750:
    swarm._update()
    i = i+1
    scores.append(swarm.best_score)
    if swarm.best_score < best_scores[-1][1]:
        best_scores.append((i, swarm.best_score))
        print_best_particle(best_scores[-1])
f.write("\n------------\n")
f.flush()
# Test...

fig, ax = plt.subplots()
ax.plot(range(len(scores)),scores )

ax.set(xlabel='Iteration)', ylabel='Loss ',
       title='Keras PSO loss plot')
ax.grid()
#ax.set_yscale('log')
ax.set_yticks(np.arange(0, 0.5, step=0.02))

fig.savefig("keraspsolog1.png")



best_weights = vector_to_weights(swarm.g, shape)


model_s.set_weights(best_weights)
p_train_score = model_s.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
p_test_score = model_s.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print("PSO -- train: {:.4f},{:.4f}  test: {:.4f},{:.4f}"
          .format(p_train_score[0], p_train_score[1], p_test_score[0], p_test_score[1]))
f.write("PSO -- train: {:.4f},{:.4f}  test: {:.4f},{:.4f}"
          .format(p_train_score[0], p_train_score[1], p_test_score[0], p_test_score[1]))

plt.show()
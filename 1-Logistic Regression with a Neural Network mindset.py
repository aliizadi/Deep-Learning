import numpy as np
import matplotlib.pyplot as plt

train_set_x_org=np.zeros()
train_set_y =np.zeros()
test_set_x_orig=np.zeros()
test_set_y=np.zeros()
classes= np.zeros()

index = 25
plt.imshow(train_set_x_org[index])

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_org.shape[1]

train_set_x_flatten = train_set_x_org.reshape(train_set_x_org.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z):
    s =  1 / (1 + np.exp(-z))
    return s


def initialized_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]

    #forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = ( -1 / m ) * np.sum( Y * np.log(A) + (1 - Y)*(np.log(1-A)))

    #backward propagation
    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y)

    assert ( dw.shape == w.shape)
    assert ( dw.type == float)
    cost = np.squeeze(cost)
    assert ( cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i%100 == 0:
            costs.append(cost)

        if i % 100 == 0:
            print( cost)

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w, b, X):

    m = X.shape[1]
    Y_predction = np.zeros((1,m))
    w= w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b )

    for i in range(A.shape[1]):
        Y_predction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_predction.shape == (1, m))

    return Y_predction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5):

    w, b = initialized_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    w = parameters["w"]
    b = parameters["b"]


    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_predction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate":learning_rate,
         "num_iterations": num_iterations}

    return d



d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005)

index = 5
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


learning_rates = [0.01, 0.001, 0.0001]
models = {}

for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i)

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label = str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.9')
plt.show()

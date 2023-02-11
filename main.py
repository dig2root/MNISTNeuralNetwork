import tensorflow as tf
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

ratios = []
n = NeuralNetwork()

for i in range(len(x_train)):
    n.train(x_train[i], y_train[i])

score = 0
for i in range(len(x_test)):
    result = n.test(x_test[i])
    if result == y_test[i]:
        score += 1
    
ratios.append(score/len(x_test))
print("Score : " + str(score))
print("Total : " + str(len(x_test)))
print("Ratio : " + str(score/len(x_test)))

print(str(ratios))
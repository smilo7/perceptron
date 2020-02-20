#simple perceptron

import numpy as np

class Perceptron:

	def __init__(self, inputs, labels, bias=-2, epochs=100, l_rate=0.01):
		self.inputs = inputs
		self.labeled_outputs = labels
		self.bias = bias
		self.epochs = epochs
		self.l_rate = l_rate
		self.weights = np.random.rand(1,2) #numbers from 0 to 1

	def predict(self, each_input):
		return np.dot(self.weights[:], each_input) + self.bias

		#step function
	def activation(self, inp):
		prediction = self.predict(inp)
		if prediction > 0:
			output = 1
		else:
			output = 0
		return output

	def train(self):
		for cycle in range(self.epochs):
			for each_inp, each_out in zip(self.inputs, self.labeled_outputs):
				#print("HELLOO", each_inp)
				#prediction = self.predict(each_inp)
				
				self.weights[:] += self.l_rate * (each_out - self.activation(each_inp)) * each_inp



AND_INPUTS = np.array([[0,0], [1,0], [0,1], [1,1]])

AND_OUTPUTS = [0,0,0,1]

p = Perceptron(AND_INPUTS, AND_OUTPUTS)
print(p.weights)
p.train()


print(p.activation(np.array([0,0])))
print(p.activation(np.array([1,0])))
print(p.activation(np.array([0,1])))
print(p.activation(np.array([1,1])))
#simple perceptron

import numpy as np

class Perceptron:

	def __init__(self, no_inputs, inputs, labels, bias=-2, epochs=10, l_rate=0.1):
		self.inputs = inputs
		self.labeled_outputs = labels
		self.bias = bias
		self.epochs = epochs
		self.l_rate = l_rate
		self.weights = np.random.rand(1, no_inputs) # randomnumbers from 0 to 1

	#predict the output based on the dot product of weights to input
	def predict(self, each_input):
		#print(self.weights[:])
		#print(each_input)
		return np.dot(self.weights[:], each_input) + self.bias

	#step function
	def activation(self, inp):
		prediction = self.predict(inp)
		if prediction > 0:
			output = 1
		else:
			output = 0
		return output

	#train the perceptron!
	def train(self):
		for cycle in range(self.epochs):
			#print("SUPER",self.inputs)
			for each_inp, each_out in zip(self.inputs, self.labeled_outputs):
				#print("HELLOO", each_inp)
				#prediction = self.predict(each_inp)
				
				self.weights[:] += self.l_rate * (each_out - self.activation(each_inp)) * each_inp



INPUTS = np.array([[0,0], [1,0], [0,1], [1,1]])

AND_OUTPUTS = [0,0,0,1]

OR_OUTPUTS = [0,1,1,1]


NOT_INPUTS = np.array([ [0], [1] ])

NOT_OUTPUTS = np.array([[1], [0]])




def test_all_inputs(no_inputs, inputs, output_labels, bias=-2):
	perceptron = Perceptron(no_inputs, inputs, output_labels, bias)
	perceptron.train()

	for each in INPUTS:
		print(perceptron.activation(each))

print("AND")
test_all_inputs(2, INPUTS, AND_OUTPUTS)


print("\nOR")
test_all_inputs(2, INPUTS, OR_OUTPUTS, -1)


print("\nNOT")
#test_all_inputs(1, NOT_INPUTS, NOT_OUTPUTS, -1)

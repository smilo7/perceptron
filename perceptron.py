'''
simple feed forward neural network
'''

import numpy as np

#input to the perceptron will take the form of 1/4 of these possible inputs
AND_INPUTS = [[0,0], [1,0], [0,1], [1,1]]

AND_OUTPUTS = [0,0,0,1]

class Perceptron:
    
	def __init__(self, inputs, output, bias=np.random.rand(), num_epochs=100, w_change=0.1):
		self.input = inputs
		self.output = output
		self.bias = bias
		self.weights = np.random.randn(1,2)*2 - 1 #make initial weights for neurons
		self.num_epochs = num_epochs
		self.w_change = w_change

	#(calcultes output before it is passed into the step function)
	def transfer_function(self, inp):
		print(inp[:])
		print(self.weights[:])
		prediction = 0
		for e, node in zip(inp, self.weights):
			prediction = np.dot(self.weights, e + self.bias)

		return prediction
        

	def predict(self, inp):
		return self.step(np.dot(self.weights, inp[:] + self.bias))

    #step function
	def step(self, prediction):
	    if (prediction > 0):
	    	return 1
	    else:
	    	return 0

	def train(self):
		for t in range(self.num_epochs):

			for inp,out in zip(self.input, self.output):
				raw_prediction = self.transfer_function(inp)
				prediction = self.step(raw_prediction)

				#error = (out - prediction)**2 * self.w_change

				error = 1/2 * (raw_prediction - out)**2 #squared mean
				self.weights[:]
				#self.back_prop(prediction, raw_prediction, out, error) # use this to try train the network
				
				#self.weights += error
				#self.bias = error * self.w_change

	def back_prop(self, prediction, raw_prediction, actual_value, error):
		if prediction != actual_value: #if prediction is not correct, then we need to change some things
			
			if prediction == 1: #shift it down by the loss function
				self.weights -= actual_value/error
			else:
				self.weights += actual_value/error



def predict_for_each():
	predict = []
	for each_input, actual_out in zip(AND_INPUTS, AND_OUTPUTS):
		predict.append(p.predict(each_input))
	return predict


def train_to_perfection():
	epochs = 1

	predic = predict_for_each()
	while predic != AND_OUTPUTS:
		p.train()
		epochs = epochs + 1
		predic = predict_for_each()
		print(predic, AND_OUTPUTS)
		#print(p.weights)


	print("finished")
	print(p.weights)


p = Perceptron(AND_INPUTS, AND_OUTPUTS, -3)
print(p.weights)

p.train() # initial training

#print(p.weights)
train_to_perfection()




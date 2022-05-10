# Include libraries which may use in implementation
from matplotlib import image as img
import time
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import glob
import numpy as np
import json
import sklearn.datasets as ds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json  # for saving the model
from numpy import save,load #for saving and loading numpy arrays
import os
from sklearn.metrics import confusion_matrix


print("This is commit")
# Create a Neural_Network class
np.random.seed(5)

class Neural_Network(object):        
	def __init__(self, no_of_layers=3, input_dim=784, neurons_per_layer =[128,64,10],activation_function=''):        
		#__init__(self, inputSize = 784,hiddenlayer1 = 128,hiddenlayer2 = 64, outputSize = 10 ,activation_function=''):
		# size of layers  #no_of_layers, input_dim, neurons_per_layer
		self.inputSize = input_dim
		 
		self.hiddenLayer1 = neurons_per_layer[0] #hiddenlayer1
		self.hiddenLayer2 = neurons_per_layer[1] #hiddenlayer2
		self.outputSize = neurons_per_layer[2]#outputSize
		self.activation_function=activation_function
		#weights
		self.W1 = np.random.randn(self.hiddenLayer1 , self.inputSize)  # randomly initialize W1 using random function of numpy
		self.b1 = np.zeros((self.hiddenLayer1, 1))

		self.W2 = np.random.randn(self.hiddenLayer2 , self.hiddenLayer1)  # randomly initialize W2 using random function of numpy
		self.b2 = np.zeros((self.hiddenLayer2, 1))
		   
		self.W3 = np.random.randn(self.outputSize , self.hiddenLayer2)  # randomly initialize W3 using random function of numpy
		self.b3 = np.zeros((self.outputSize, 1))
	def feedforward(self, X):
		#forward propagation through our network
		# dot product of X (input) and set of weights
		# apply activation function (i.e. whatever function was passed in initialization)
		##########       Layer 1
		
		z1 = np.dot(self.W1, X) + self.b1

		if self.activation_function == 'sigmoid':
			a1=self.sigmoid(z1)

		if self.activation_function == 'tanh':
			a1 = self.tanh(z1)
		
		if self.activation_function == 'relu':
			a1 = self.relu(z1)
		############      Layer2
		z2 = np.dot(self.W2, a1) + self.b2
		if self.activation_function == 'sigmoid':
			a2 = self.sigmoid(z2)
		if self.activation_function == 'tanh':
			a2 = self.tanh(z2)
		
		if self.activation_function == 'relu':
			a2 = self.relu(z2)
		############      Layer2
		z3 = np.dot(self.W3, a2) + self.b3
		a3 = self.softmax(z3)

		#cache the activations for backpropagation
		cache = {"z1": z1,
				 "a1": a1,
				 "z2": z2,
				 "a2": a2,
				 "z3": z3,
				 "a3": a3}
	
		return  a3, cache   # return your answer with as a final output of the network

	def sigmoid(self, s):
		# activation function
		return 1/(1 + np.exp(-s)) # apply sigmoid function on s and return it's value

	def sigmoid_derivative(self, s):
		#derivative of sigmoid
		return self.sigmoid(s)*(1-self.sigmoid(s))# apply derivative of sigmoid on s and return it's value 
	
	def tanh(self, s):
		# activation function
		return (np.exp(s) - np.exp(-s))/(np.exp(s) + np.exp(-s)) # apply tanh function on s and return it's value

	def tanh_derivative(self, s):
		#derivative of tanh
		return 1 - np.power(self.tanh(s),2) # apply derivative of tanh on s and return it's value
	
	def relu(self, s):
		# activation function
		return np.maximum(s, 0.0) # apply relu function on s and return it's value

	def relu_derivative(self, s):
		#derivative of relu
		return (s>0) * 1 # apply derivative of relu on s and return it's value
	def softmax(self,X):
		expo = np.exp(X)
		expo_sum = np.sum(np.exp(X),axis=0)
		return expo/expo_sum

	def backwardpropagate(self,X, Y, y_pred, lr, cache):
		# backward propagate through the network
		# compute error in output which is loss compute cross entropy loss function
		# applying derivative of that applied activation function to the error
		# adjust set of weights

		#load the values that have been cached during forward propagation
		a1 = cache["a1"]
		a2 = cache["a2"]
		a3 = cache["a3"]
		z1 = cache["z1"]
		z2 = cache["z2"]
		z3 = cache["z3"]
		
		m = X.shape[1]
		# Backward propagation: calculate dW1, db1, dW2, db2. 

		dz3 = y_pred
		dz3[Y, range(m)] -= 1
		dz3 = dz3/m
		
		dW3 = (1/m)*np.dot(dz3,a2.T)
		db3 = (1/m)*(np.sum(dz3, axis=1,keepdims=True))

		da2 = np.dot(self.W3.T,dz3)

		if self.activation_function == 'sigmoid':
			dz2 = np.multiply(da2,self.sigmoid_derivative(z2))

		if self.activation_function == 'tanh':
			dz2 = np.multiply(da2,self.tanh_derivative(z2))
		
		if self.activation_function == 'relu':
			dz2 = np.multiply(da2,self.relu_derivative(z2))
		
		
		dW2 = (1/m)*np.dot(dz2,a1.T)

		db2 = (1/m)*(np.sum(dz2, axis=1,keepdims=True))
		da1 = np.dot(self.W2.T,dz2)

		if self.activation_function == 'sigmoid':
			dz1 = np.multiply(da1,self.sigmoid_derivative(z1))

		if self.activation_function == 'tanh':
			dz1 = np.multiply(da1,self.tanh_derivative(z1))
		
		if self.activation_function == 'relu':
			dz1 = np.multiply(da1,self.relu_derivative(z1))
		
		dW1 = (1/m)*np.dot(dz1, X.T)
		db1 = (1/m)*(np.sum(dz1, axis=1,keepdims=True))

		#update step
		#Update the weights
		self.W1 = self.W1 - lr * dW1
		self.b1 = self.b1 - lr * db1
		self.W2 = self.W2 - lr * dW2
		self.b2 = self.b2 - lr * db2
		self.W3 = self.W3 - lr * dW3
		self.b3 = self.b3 - lr * db3
		
		
		return
	# Function to load dataset

	def loadDataset(self,path):
		path = path
		print(path)
		print('Loading Dataset...')
		train_x, train_y, test_x, test_y = [], [], [], []
		for i in range(10):
			print(f'Loading images form {path} /train/ + {str(i)}')
			for filename in glob.glob(path + '/train/' + str(i)+'/*.png'):
				im=img.imread(filename)
				train_x.append(im)
				train_y.append(i)
		for i in range(10):
			print(f'Loading images form {path} /test/ + {str(i)}')
			for filename in glob.glob(path + '/test/' + str(i)+'//*.png'):
				im=img.imread(filename)
				test_x.append(im)
				test_y.append(i)
		print('Dataset loaded...')
		return np.array(train_x), np.array(train_y),np.array(test_x),np.array(test_y)

	def meanSubtraction(self,data):
		mean_of_data=np.mean(data, axis=0)
		data = np.subtract(data,mean_of_data)
		return data

	def tSNE(self,X,Y,plot_title):
		feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
		df = pd.DataFrame(X,columns=feat_cols)
		df['y'] = Y
		df['label'] = df['y'].apply(lambda i: str(i))
		X, y = None, None
		print('Size of the dataframe: {}'.format(df.shape))

		# For reproducability of the results
		np.random.seed(42)
		rndperm = np.random.permutation(df.shape[0])

		N = 10000
		df_subset = df.loc[rndperm[:N],:].copy()
		data_subset = df_subset[feat_cols].values
		time_start = time.time()
		tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
		tsne_results = tsne.fit_transform(data_subset)
		print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

		df_subset['tsne-2d-one'] = tsne_results[:,0]
		df_subset['tsne-2d-two'] = tsne_results[:,1]
		plt.figure(figsize=(16,10))
		sns.scatterplot(
			x="tsne-2d-one", y="tsne-2d-two",
			hue="y",
			palette=sns.color_palette("hls", 10),
			data=df_subset,
			legend="full",
			alpha=0.3
		)
		plt.title(plot_title)
		plt.show()
	
	def crossentropy(self, Y, Y_pred):
			
		m = Y.shape[1]
		
		# compute error based on crossentropy loss 
		log_likelihood = -np.log(Y_pred[Y,range(m)])
		#print("log_liklyhood",log_likelihood)
		loss = np.sum(log_likelihood) / m
		return loss
		
		#train_SGD(trainX, trainY, validationX = validX, validationY = validY,learningRate = 2, batch_size=50, training_epochs=50, plot_err= True)
	def train_SGD(self, trainX, trainY, validationX, validationY, learningRate, batch_size, training_epochs, plot_err= True):
		# feed forward trainX and trainY and recivce predicted value
		# backpropagation with trainX, trainY, predicted value and learning rate.
		# if validationX and validationY are not null than show validation accuracy and error of the model by printing values.
		# plot error of the model if plot_err is true
		
		model = self
		lr = learningRate
		epochs = training_epochs
		batch_size=batch_size
		n_examples = trainX.shape[1]
		epoch_loss = []
		epoch_accuracy =[]
		validation_loss = []
		validation_accuracy = []
		print("\n\nTraining...")
		for epoch in range(epochs):
			loss=0
			#shuffle the data (SGD)
			trainX, trainY = shuffle(trainX.T, trainY.T, random_state=0)
			trainX,trainY = trainX.T,trainY.T
			n_train = trainX.shape[1]
			
			#create mini batches
			Xmini_batches = np.array([trainX[:, k: k+batch_size] for k in range(0, n_train, batch_size)])
			Ymini_batches = np.array([trainY[:, k: k+batch_size] for k in range(0, n_train, batch_size)])
			
			for Xmini_batch,Ymini_batch in zip(Xmini_batches,Ymini_batches):
				y_hat,cache=model.feedforward(Xmini_batch)
				loss = model.crossentropy(Ymini_batch, y_hat)
				acc= model.accuracy(Ymini_batch, y_hat)
				model.backwardpropagate(Xmini_batch, Ymini_batch, y_hat, lr,cache)
			
			
			epoch_loss.append(loss)
			epoch_accuracy.append(acc)
			# validation
			if validationX is not None:
				y_hat_valid,_ = model.feedforward(validationX)
				valid_loss = model.crossentropy(validationY,y_hat_valid)
				valid_acc= model.accuracy(validationY,y_hat_valid)
				validation_loss.append(valid_loss)
				validation_accuracy.append(valid_acc)
				print("Epoch = %3d   Training Loss = %3.3f  Validation Loss = %3.3f"%(epoch, epoch_loss[epoch],validation_loss[epoch]) )
			else:
				print("Epoch = %3d   Training Loss = %3.3f"%(epoch, epoch_loss[epoch]) )
		print("\nDone.")  
		if plot_err:
			plt.figure()
			plt.plot(np.linspace(0,epochs,epochs),epoch_loss)
			plt.plot(np.linspace(0,epochs,epochs),validation_loss)
			plt.legend(["training loss", "validation loss"], loc ="upper right")
			plt.title(f'Learning Rate={lr}   #epoches={epochs}    batch Size={batch_size}')
			plt.figure()
			plt.plot(np.linspace(0,epochs,epochs),epoch_accuracy)
			plt.plot(np.linspace(0,epochs,epochs),validation_accuracy)
			plt.legend(["training accuracy", "validation accuracy"], loc ="lower right")
			plt.title(f'Learning Rate={lr}   #epoches={epochs}    batch Size={batch_size}')
			plt.show()  

		return model

	def predict(self, testX):
		# predict the value of testX
		a3,_=self.feedforward(testX)
		
		return a3  #predictions

	def predict_digit(self,testX):
		a3,_=self.feedforward(testX)
		pred = np.where(a3 > 0.5, 1, 0).T
		return np.argmax(pred,axis=1)

	def accuracy(self, real_val,pred):
		
		# predict the value of trainX
		# compare it with testY
		# compute accuracy, print it and show in the form of picture
		pred = np.where(pred > 0.5, 1, 0).T
		acc = np.sum(np.equal(real_val, np.argmax(pred,axis=1))) / (real_val.shape[1])
		return acc # return accuracy    
		
	def saveModel(self,name):
		# save your trained model, it is your interpretation how, which and what data you store
		# which you will use later for prediction
		data = {
			'sizes' : [self.inputSize,self.hiddenLayer1, self.hiddenLayer2, self.outputSize],
			'weights': [self.W1.tolist(), self.W2.tolist(), self.W3.tolist()],
			'biases' : [self.b1.tolist(), self.b2.tolist(), self.b3.tolist()],
			'activation' :[self.activation_function]

		}
		
		f = open('models/'+name,'w')
		json.dump(data,f)
		f.close()
		return 1

		
	def loadModel(self,name):
		# load your trained model, load exactly how you stored it.
		f=open('models/'+name)
		data = json.load(f)

		self.inputSize,self.hiddenLayer1, self.hiddenLayer2, self.outputSize = data['sizes']
		
		self.W1, self.W2, self.W3 = data['weights']
		
		self.b1, self.b2, self.b3 = data['biases']

		self.activation_function = data['activation'][0]
 
		f.close()
		
		return 1

	def confusion_Mat(self, y_true, y_pred,name):
		pred = np.where(y_pred > 0.5, 1, 0).T
		pred = np.expand_dims(np.argmax(pred,axis=1),axis=1).T
		classes = ('0', '1', '2', '3', '4','5', '6', '7', '8', '9')
		cf_matrix = confusion_matrix(y_true.T, pred.T)
		df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],columns = [i for i in classes])
		plt.figure(figsize = (12,7))
		sns.heatmap(df_cm, annot=True).set(title=name)
		plt.show()

	def one_Hot_encode(self, arr):
		
		b = np.zeros((arr.size, arr.max()+1))
		b[np.arange(arr.size),arr] = 1
		return b

def main():   
	

	wd_path = os.getcwd()
	data_path = 'Task3_MNIST_Data/'
	

	_path=os.path.join(wd_path,data_path) 
	
				
	model = Neural_Network(3,784,[128,64,10],activation_function='sigmoid') #relu doesn't work
	
	
	#i load the mnist images of handwritten digits and store it in numpy array 
	#loading images takes much longer time then loading numpy arrays
	##################################################################################
	#      saving the data to numpy array and save in disk to reduce the data loading time
	#      numpy array takes less time to load
	# Uncomment the below code while running the first time.

	
	trainX,trainY,testX,testY = model.loadDataset(_path)
	# save to npy file
	save('data_arrays/xtrain.npy', trainX)
	save('data_arrays/ytrain.npy', trainY)
	save('data_arrays/xtest.npy', testX)
	save('data_arrays/ytest.npy', testY)
	

	#loading the data array
	trainX = load('data_arrays/xtrain.npy')
	trainY = load('data_arrays/ytrain.npy')
	testX = load('data_arrays/xtest.npy')
	testY = load('data_arrays/ytest.npy')

	

	trainX = trainX.reshape((trainX.shape[0],28*28))
	testX = testX.reshape((testX.shape[0],28*28))

	##################################################################
	##   Mean subtraction   #
	##################################################################
	
	
	data = model.meanSubtraction(np.vstack((trainX,testX)))          
	trainX,testX=data[0:60000,:],data[60000:70000,:]
	

	##################################################################
	##   shuffeling the data and dividing to train validation sets   #
	##################################################################
	
	#Note: shuffle this dataset before dividing it into three parts
	
	trainX, trainY = shuffle(trainX, trainY, random_state=0)
	
	trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size=0.1, random_state=0)

	trainX = np.transpose(trainX)# training data point  (2,640)
	trainY = np.transpose(trainY)# training lables      (1,640)
	trainY = np.expand_dims(trainY, axis=0)

	validX = np.transpose(validX) # validation data point
	validY = np.transpose(validY) # validation lables
	validY = np.expand_dims(validY, axis=0)

	testX = np.transpose(testX) # testing data point
	testY = np.transpose(testY)# testing lables
	testY = np.expand_dims(testY, axis=0)

	print(trainX.shape, trainY.shape,  validX.shape, validY.shape)
	
	###################################################################
	#    ONE Hot encoding
	###################################################################
	TrainY = model.one_Hot_encode(trainY)
	ValidY = model.one_Hot_encode(validY)
	TestY = model.one_Hot_encode(testY)
	#print the value for confirmation
	print("\n######################################################\n One Hot encoded vectors\n")
	print("one Hot encoded vector is = ",TrainY[1]," For the value ",trainY[0][1])  
	print("one Hot encoded vector is = ",ValidY[2]," For the value ",validY[0][2])  
	print("one Hot encoded vector is = ",TestY[1]," For the value ",testY[0][1])  
	print("\n")

	

	###################################################################
	#    Training the Model
	###################################################################
	
	model = model.train_SGD(trainX, trainY, validationX = validX, validationY = validY,learningRate = 0.5, batch_size=20, training_epochs=5, plot_err= True)
	#save the best model which you have trained, 
	model.saveModel('Task3model5.json')
	# check accuracy of that model
	trAcc,_ = model.feedforward(trainX)
	tsAcc,_ = model.feedforward(testX)
	train_accuracy = model.accuracy(trainY, trAcc) *100
	test_accuracy = model.accuracy(testY,tsAcc) *100

	print("Train accuracy", train_accuracy)
	print("Test accuracy", test_accuracy)
    
	

	###################################################################
	#    Loading the trained model and checking accuracies and confusion matrix
	###################################################################
	# create class object
	mm = Neural_Network()
	# load model which will be provided by you
	mm.loadModel('Task3model5.json')

	
	
	
	
	###############################################################
	#     tSNE plots
	#test data tsne plot 
	
	data = testX
	datay = testY

	mm.tSNE(data.T,datay.T,'Input layer tSNE')

	#layer 1 tsne plot
	_, tsnCache = mm.feedforward(data)
	tsnA1 = tsnCache['a1']
	mm.tSNE(tsnA1.T,datay.T,'First Hidden layer tSNE')

	#Layer 2 tsne plot
	tsnA2 = tsnCache['a2']
	mm.tSNE(tsnA2.T,datay.T,'Second Hidden layer tSNE')
	
	
	################################################################
	#         Confusion Matrix calculation
	#
	#training data Confusion matrix
	print("\n######################################################\n Accuracies \n")
	_pred = mm.predict(trainX)
	mm.confusion_Mat(trainY, _pred,'Training Data Confusion Matrix')
	train_acc=mm.accuracy(trainY,_pred)*100
	print("Training accuracy = ",train_acc)

	#Validation data Confusion matrix
	_pred = mm.predict(validX)
	mm.confusion_Mat(validY, _pred,'Validation Data Confusion Matrix')
	valid_acc=mm.accuracy(validY,_pred)*100
	print("Validation accuracy = ",valid_acc)

	#training data Confusion matrix
	_pred = mm.predict(testX)
	mm.confusion_Mat(testY, _pred,'Test Data Confusion Matrix')
	test_acc=mm.accuracy(testY,_pred)*100
	print("Test accuracy = ",test_acc)

	plt.bar(['train','valid', 'test'],[train_acc, valid_acc, test_acc])
	plt.xlabel('Data')
	plt.ylabel("Accuracy")
	plt.title('Train , Validation and test accuracies Bar Plot')
	plt.show()

	print("\n\n\n")
	##############################################################
	#        Predict and display a digit
	k = 9000  #max 10000
	digit= testX[:,k:k+1]
	pred_dig = mm.predict_digit(digit)
	print("Predicted digit = ",pred_dig)
	plt.imshow(digit.reshape(28,28))
	plt.show()
	print("done")
	


if __name__ == '__main__':
	main()
# Include libraries which may use in implementation
from pyexpat import model
import torch
import torchvision as tv
from torchvision import transforms
import math
import numpy
import torch.utils.data as data #At the heart of PyTorch data loading utility is torch.utils.data.DataLoader class.
								#It represents a Python iterable over a dataset
import torch.nn as nn #torch.nn provide us many more classes and modules to implement and train the neural network.
import torch.nn.functional as F
import torch.optim as optim # torch.optim is a package implementing various optimization algorithms e.g SGD, ASGD

from matplotlib import image as img
import csv
import time
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json  # for saving the model
from numpy import save,load #for saving and loading numpy arrays
import os
from sklearn.metrics import confusion_matrix



# Create a Neural_Network class
np.random.seed(5)

class Neural_Network(nn.Module):        
	def __init__(self, no_of_layers=3, input_dim=784, neurons_per_layer =[128,64,10],activation_function=''):
		super(Neural_Network, self).__init__()
		self.l1 = nn.Linear(input_dim, neurons_per_layer[0])
		self.relu = nn.ReLU()
		self.l2 = nn.Linear(neurons_per_layer[0], neurons_per_layer[1])
		self.relu = nn.ReLU()
		self.l3 = nn.Linear(neurons_per_layer[1], neurons_per_layer[2])
		
	def forward(self, x):
		x = x.view(x.shape[0],-1)
		x = self.l1(x)
		x = self.relu(x)
		x = self.l2(x)
		x = self.relu(x)
		x = self.l3(x)
		return x#F.log_softmax(x)

		

	def loadDataset(self, path, batch_size):
		path = path
		print("pathhhhh",path)
		print('Loading Dataset...')
		train_x, train_y, test_x, test_y = [], [], [], []
		fname_train=[]
		label_train=[]
		fname_test=[]
		label_test=[]
		# loading the train data
		file = open(path+'train/train.csv')
		csvreader = csv.reader(file)
		j=0
		k=5
		
		for i in csvreader:
			fname_train.append(i[0])
			label_train.append(i[1])
			#j=j+1
			#if j%2000==0:
			#	print("Loaded Files",j)
		del fname_train[0]
		del label_train[0]

		train_y = label_train
		
		for filename in fname_train:
			im=img.imread(path+'train/train_new/'+filename)
			train_x.append(im)
			j=j+1
			if j%2000==0:
				print("Loaded Train Files",j)
		
		# loading the test data
		file = open(path+'test/test.csv')
		csvreader = csv.reader(file)
		j=0
		for i in csvreader:
			fname_test.append(i[0])
			label_test.append(i[1])
			
			
		del fname_test[0]
		del label_test[0]
		print("len test label",len(label_test))
		test_y = label_test
		j=0
		for filename in fname_test:
			im=img.imread(path+'test/test_new/'+filename)
			test_x.append(im)
			j=j+1
			if j%2000==0:
				print("Loaded test Files",j)

		train_x, train_y, test_x, test_y = np.array(train_x), np.array(train_y),np.array(test_x),np.array(test_y)
		#converting the string arrays to int
		train_y = train_y.astype(np.int64)
		test_y = test_y.astype(np.int8)

		#######################################################
		#making pytorch data loaders
		#######################################################
		transform = transforms.Compose( [transforms.ToTensor(),
	 							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		train = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
		train_set, val_set = torch.utils.data.random_split(train, [36470, 7000])#[3,1])#[36470, 7000])  #total 43470
		
		train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
		vaild_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=True)
		
		#checking the split
		"""
		for i in range(5):
			print("check random = ", val_set[i][1])
			plt.imshow(val_set[i][0])
			plt.show()
		"""
		
		

		test = torch.utils.data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
		test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=True)
		
			
		print('Dataset loaded...')
		return train_loader, vaild_loader, test_loader

	def loadDatasetTorch(data_dir, training_size, validation_size, test_size, BATCH_SIZE, SHUFFLE):
	
		mean = 0.0
		variance = 0.5
	
		transform = transforms.Compose([transforms.ToTensor(),
									transforms.Grayscale(1),
									transforms.Normalize(0, math.sqrt(variance))])
		# Loading the training dataset. We need to split it into a training and validation part
		train_dataset = tv.datasets.ImageFolder(root = data_dir + '/train', transform = transform)
		train_set, val_set = torch.utils.data.random_split(train_dataset, [training_size, validation_size])
		# Loading the test dataset. We need to split it into the user define test_size
		test_set = tv.datasets.ImageFolder(root = data_dir + '/test', transform = transform)
		test_set , unused_test_set = torch.utils.data.random_split(test_set, [test_size, len(test_set)-test_size])

		# We define a set of data loaders that we can use for various purposes later.
		# Note that for actually training a model, we will use different data loaders
		# with a lower batch size.
		train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False) # Training data is kept to shuffle every time by default.
		val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE, drop_last=False)
	
		test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE, drop_last=False)
	
		return train_loader, val_loader, test_loader


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
	
		
	def train(self, train_loader, valid_loader, criterion, optimizer, training_epochs, plot_err= True):
		model = self
		trainLoss=[]
		trainAcc=[]
		validLoss=[]
		validAcc=[]
		for e in range(training_epochs):
			train_loss = 0.0
			train_acc =0.0
			for data, labels in train_loader:
				# Transfer Data to GPU if available
				if torch.cuda.is_available():
					data, labels = data.cuda(), labels.cuda()
				
				# Clear the gradients
				optimizer.zero_grad()
				# Forward Pass
				target = model(data)
				
				#print("Target++++",target,(labels[0]))
				# Find the Loss
				loss = criterion(target,labels)
				# Calculate gradients
				loss.backward()
				# Update Weights
				optimizer.step()
				# Calculate Loss
				train_acc = train_acc + torch.sum(torch.argmax(target) == labels)
				
				train_loss += loss.item()
			

			valid_loss = 0.0
			valid_acc = 0.0
			#model.eval()     # Optional when not using Model Specific layer
			for data, labels in valid_loader:
				# Transfer Data to GPU if available
				if torch.cuda.is_available():
					data, labels = data.cuda(), labels.cuda()
				
				# Forward Pass
				target = model(data)
				# Find the Loss
				loss = criterion(target,labels)
				# Calculate Loss
				valid_loss += loss.item()
				valid_acc = valid_acc + torch.sum(torch.argmax(target) == labels)
			
			trainLoss.append(train_loss/len(train_loader))
			validLoss.append(valid_loss/len(valid_loader))
			trainAcc.append(train_acc/len(train_loader))
			validAcc.append(valid_acc/len(valid_loader))

			print(f'Epoch {e+1} Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
			"""
			if min_valid_loss > valid_loss:
				print(f'Validation Loss Decreased({min_valid_loss:.6f\
				}--->{valid_loss:.6f}) \t Saving The Model')
				min_valid_loss = valid_loss
				
				# Saving State Dict
				torch.save(model.state_dict(), 'saved_model.pth') 
			"""
		return model,trainLoss, validLoss, trainAcc, validAcc

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
		pass

		
	def loadModel(self,name):
		pass

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
	data_path = 'Data/'
	

	_path=os.path.join(wd_path,data_path)
	
	model = Neural_Network(3,784,[128,64,10],activation_function='relu') #relu doesn't work
	

	train_loader, valid_loader, test_loader=model.loadDataset(_path,batch_size=64) 			
	
	print(model)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
	model, trainLoss, validLoss, trainAcc, validAcc=model.train(train_loader, valid_loader, criterion, optimizer, training_epochs = 10, plot_err= True)
	plt.figure()
	plt.plot(trainLoss)
	plt.plot(validLoss)
	plt.figure()
	plt.plot(trainAcc)
	plt.plot(validAcc)
	plt.show()
	
	##################################################################
	##   Mean subtraction   #
	##################################################################
	
	
	###################################################################
	#    Training the Model
	###################################################################
	"""
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
	"""
	

	###################################################################
	#    Loading the trained model and checking accuracies and confusion matrix
	###################################################################
	"""
	# create class object
	mm = Neural_Network()
	# load model which will be provided by you
	mm.loadModel('Task3model5.json')

	"""
	
	
	
	###############################################################
	#     tSNE plots
	#test data tsne plot 
	"""
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
	"""
	
	################################################################
	#         Confusion Matrix calculation
	#
	#training data Confusion matrix
	"""
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
	"""
	##############################################################
	#        Predict and display a digit
	"""
	k = 9000  #max 10000
	digit= testX[:,k:k+1]
	pred_dig = mm.predict_digit(digit)
	print("Predicted digit = ",pred_dig)
	plt.imshow(digit.reshape(28,28))
	plt.show()
	print("done")
	"""


if __name__ == '__main__':
	main()
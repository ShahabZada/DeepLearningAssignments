# %%

import torch
import torchvision as tv
from torchvision import transforms
import torch.utils.data as data 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 

import random
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
from numpy import save,load #for saving and loading numpy arrays
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



# Create a Neural_Network class
np.random.seed(5)
# %%
class Neural_Network(nn.Module):        
	def __init__(self):
		super(Neural_Network, self).__init__()
		
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)
	def forward(self, x):
		x = self.conv1(x)
		x = F.max_pool2d(x, 2)
		#x = F.tanh(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = self.conv2_drop(x)
		x = F.max_pool2d(x, 2)
		#x = F.tanh(x)
		x = F.relu(x)
		x = x.view(-1, 320)
		x = self.fc1(x)
		#x = F.tanh(x)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return x

		

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

		train_x, train_y, test_x, test_y = np.expand_dims(np.array(train_x), axis=1), np.array(train_y), np.expand_dims(np.array(test_x), axis=1) ,np.array(test_y)
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
		test_loader = torch.utils.data.DataLoader(test, 10000, shuffle=False) #making one batch of whole test data
		
			
		print('Dataset loaded...')
		return train_loader, vaild_loader, test_loader

	


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
	
		
	def train(self, train_loader, valid_loader, criterion, optimizer,lr_decay_scheduler, training_epochs):
		model = self
		trainLoss=[]
		trainAcc=[]
		validLoss=[]
		validAcc=[]
		min_valid_loss = 4.0
		for e in range(training_epochs):
			train_loss = 0.0
			train_acc =0.0
			total=0.0
			correct=0.0
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
				_, predicted = target.max(1)
				total += labels.size(0)
				correct += predicted.eq(labels).sum().item()
				
				#train_acc = train_acc + torch.sum(torch.argmax(target) == labels)
				
				train_loss += loss.item()
			
			
			
			trainLoss.append(train_loss/len(train_loader))
			trainAcc.append(100.*correct/total)

			valid_loss = 0.0
			valid_acc = 0.0
			total=0.0
			correct=0.0
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
				_, predicted = target.max(1)
				total += labels.size(0)
				correct += predicted.eq(labels).sum().item()
			
			
			validLoss.append(valid_loss/len(valid_loader))
			validAcc.append(100.*correct/total)
			
			if lr_decay_scheduler is not None:
				lr_decay_scheduler.step()
			min_valid_loss = model.EarlyStopping(min_valid_loss, valid_loss/len(valid_loader))

			print(f'Epoch {e+1} Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
			
			
			
		return model,trainLoss, validLoss, trainAcc, validAcc

	def EarlyStopping(self,min_valid_loss, valid_loss):
		min_valid_loss =min_valid_loss
		if min_valid_loss > valid_loss:
			model=self
			print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
			min_valid_loss = valid_loss
			
			# Saving State Dict
			torch.save(model.state_dict(), 'saved_model.pth') 
		
		return min_valid_loss

	def predict(self, testX):
		# predict the value of testX
		test_pred=self(testX)
		
		
		return test_pred  #predictions

	def predict_digit(self,testX):
		test_pred=self(testX)
		_, predicted = test_pred.max(1)
		return predicted

	def accuracy(self, labels,test_pred):
		_, predicted = test_pred.max(1)
		
		total = labels.size(0)
		
		correct = predicted.eq(labels).sum().item()
		
		Acc = (100.*correct/total)	
		
		return Acc # return accuracy    
		
	def saveModel(self,name):
		pass

		
	def loadModel(self,name):
		pass

	def confusion_Mat(self, y_true, y_pred,name):
		_, pred = y_pred.max(1)
		
		classes = ('0', '1', '2', '3', '4','5', '6', '7', '8', '9')
		cf_matrix = confusion_matrix(y_true, pred)
		df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],columns = [i for i in classes])
		plt.figure(figsize = (12,7))
		sns.heatmap(df_cm, annot=True).set(title=name)
		plt.show()

	def precision_recall_f1score(self, y_true, y_pred):
		_, pred = y_pred.max(1)
		
		print(classification_report(y_true, pred, digits=10))
		#return (Acc_score, prec_score, rec_score, F1_score)
# %%
	

#def main():   
	
# %%
wd_path = os.getcwd()
data_path = 'Data/'


_path=os.path.join(wd_path,data_path)

#Initialize the model class (All modeules and functions are in one class)
model = Neural_Network()

batch_size=64

##################################################################
##   Loading the data (see the loadDataset function)            ##
##################################################################
train_loader, valid_loader, test_loader=model.loadDataset(_path,batch_size = batch_size) 			

print(model)
# %%

##################################################################
##              Important parameters for the model              ##
##################################################################
plot_err = True
epochs = 2
learning_rate = 0.05
batch_size =64
#####################################
#Loss function and optimizer
model = Neural_Network()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#criterion = nn.focalLoss()
optimizer = optim.Adam(model.parameters(),lr=0.008,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)
##################################################################
##                        Learning rate decay                   ##
##################################################################
decayRate = 0.96
lr_decay_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

##################################################################
##                        Training The model                    ##
##################################################################
# The EarlyStoping is implemented in the class neural network
#  
model, trainLoss, validLoss, trainAcc, validAcc=model.train(train_loader, valid_loader, criterion, optimizer=optimizer, lr_decay_scheduler=lr_decay_scheduler, training_epochs = epochs)

#saving the model
torch.save(model.state_dict(),'saved_model.pth') 
# %%

##################################################################
##   plotting the train and validation errors and accuracies   #
##################################################################
if plot_err:
		plt.figure()
		plt.plot(np.linspace(0,epochs,epochs),trainLoss)
		plt.plot(np.linspace(0,epochs,epochs),validLoss)
		plt.legend(["training loss", "validation loss"], loc ="upper right")
		plt.title(f'Learning Rate={learning_rate}   #epoches={epochs}    batch Size={batch_size}')
		plt.figure()
		plt.plot(np.linspace(0,epochs,epochs),trainAcc)
		plt.plot(np.linspace(0,epochs,epochs),validAcc)
		plt.legend(["training accuracy", "validation accuracy"], loc ="lower right")
		plt.title(f'Learning Rate={learning_rate}   #epoches={epochs}    batch Size={batch_size}')
		plt.show()  



# %%
###################################################################
#    Loading the trained model and checking accuracies and confusion matrix
###################################################################

# create class object

trainedModel = model
trainedModel.load_state_dict(torch.load('task2_model.pth'))

###############################################################
#                  test data analysis
###############################################################
testAcc = 0
tdata = 0
tlabel=0
for data, labels in test_loader:
	tdata=data
	tlabel=labels
################	
#predict a digit
################
test_pred = trainedModel.predict(tdata)
testAcc = trainedModel.accuracy(labels,test_pred)
print(trainedModel.predict_digit(tdata[0]))

################
# Accuracy
################
print("test Accuracy = ",testAcc)

################
# precision Recall and F1-score
################

print(trainedModel.precision_recall_f1score(labels, test_pred))


################
#Confusion Matrix calculation
################

trainedModel.confusion_Mat(tlabel, test_pred,'Test Data Confusion Matrix')

# %%
###############################################################
#       ploting correct and wrong predicted digits
###############################################################
test_pred = trainedModel(tdata)
_, predicted = test_pred.max(1)
all_pred = predicted.eq(labels)
wrong_pred = torch.where(all_pred==False)

correct_pred = torch.where(all_pred==True)
print(correct_pred)
print(wrong_pred)
print(all_pred)

fig = plt.figure(figsize=(10, 10))
rows , columns = 2,2

for i in range(1,5):
	fig.add_subplot(rows, columns, i)

	n = random.randint(0,len(wrong_pred[0]))

	# showing image
	plt.imshow(tdata[wrong_pred[0][n].item()][0])
	plt.axis('off')
	plt.title(f'correct label = {predicted[wrong_pred[0][n].item()]} but predicted digit is')
plt.savefig("wrong.jpg")
fig = plt.figure(figsize=(10, 10))
for i in range(1,5):
	fig.add_subplot(rows, columns, i)

	n = random.randint(0,len(correct_pred[0]))

	# showing image
	plt.imshow(tdata[correct_pred[0][n].item()][0])
	plt.axis('off')
	plt.title(f'correct label = {predicted[correct_pred[0][n].item()]} predicted digit is')

plt.savefig("correct.jpg")

# %%
###############################################################
#       Visualization
###############################################################
model_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # we will save the 49 conv layers in this list
# get all the model children as list
model_children = list(trainedModel.children())
print(model_children)


# counter to keep count of the conv layers
counter = 0 

# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolutional layers: {counter}")

# take a look at the conv layers and the respective weights
for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
# visualize the first conv layer filters
plt.figure(figsize=(20, 20))
for i, filter in enumerate(model_weights[1]):
    plt.subplot(10, 10, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    plt.imshow(filter[0, :, :].detach(), cmap='gray')
    plt.axis('off')
    plt.savefig('filter.png')
plt.show()
# %%
#######################
# visualizing output of final convolutional layer
#######################
outputs= []
def hook(module, input, output):
    outputs.append(output)


#trainedModel.layer4[0].conv2.register_forward_hook(hook)
#out = res50_model(res)
#out = res50_model(res1)
#print(outputs)
# %%
###############################################################
#     tSNE plots
#test data tsne plot 

data = tdata.reshape(tdata.shape[0], -1)
data_pred = predicted


trainedModel.tSNE(data,data_pred,'test data tsne plot')



#if __name__ == '__main__':
#	main()


# %%

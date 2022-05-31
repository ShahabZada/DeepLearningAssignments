# %%

import torch
import torchvision as tv
from torchvision import transforms
import torch.utils.data as data 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset
from torchsummary import summary
from torch.autograd import Variable

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
from collections import OrderedDict


np.random.seed(5)
# %%
class myMNISTdata(Dataset):
	
	def __init__(self, csv_file, root_dir, transform=None):
		self.datacsv = pd.read_csv(root_dir + csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.datacsv)

	def __getitem__(self, index):

		img_path = os.path.join(self.root_dir, self.datacsv.iloc[index, 0])
		image=img.imread(img_path)
		label = torch.tensor(int(self.datacsv.iloc[index, 1]))
		
		if self.transform:
			image = self.transform(image)

		return (image, label)

# %%


class ConvDw(nn.Module):
	def __init__(self, inChannels, stride=(1, 1)):
		super(ConvDw, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(inChannels, inChannels, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=inChannels),
			nn.BatchNorm2d(inChannels),
			nn.ReLU(inplace=True)
		)

	def forward(self, input_image):
		x = self.conv(input_image)
		return x


class ConvPw(nn.Module):
	def __init__(self, inChannels, outChannels):
		super(ConvPw, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(inChannels, outChannels, kernel_size=(1, 1)),
			nn.BatchNorm2d(outChannels),
			nn.ReLU(inplace=True)
		)

	def forward(self, input_image):
		x = self.conv(input_image)
		return x


class MobileNet_block(nn.Module):
	def __init__(self, inChannels, outChannels, stride=(1, 1)):
		super(MobileNet_block, self).__init__()
		self.dw = ConvDw(inChannels=inChannels, stride=stride)
		self.pw = ConvPw(inChannels=inChannels, outChannels=outChannels)

	def forward(self, input_image):
		x = self.pw(self.dw(input_image))
		return x


class MobileNet(nn.Module):
	def __init__(self, inChannels=1, num_filter=32, num_classes=10):
		super(MobileNet, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(inChannels, num_filter, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
			nn.BatchNorm2d(num_filter),
			nn.ReLU(inplace=True)
		)

		self.inChannels = num_filter

		self.model = nn.Sequential(
			MobileNet_block(32, 64, 1),
			MobileNet_block(64, 128, 2),
			MobileNet_block(128, 128, 1),
			MobileNet_block(128, 256, 2),
			MobileNet_block(256, 256, 1),
			MobileNet_block(256, 512, 2),
			MobileNet_block(512, 512, 1),
			MobileNet_block(512, 512, 1),
			MobileNet_block(512, 512, 1),
			MobileNet_block(512, 512, 1),
			MobileNet_block(512, 512, 1),
			MobileNet_block(512, 1024, 2),
			MobileNet_block(1024, 1024, 1),
		)

		self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
		self.fc = nn.Sequential(
			nn.Linear(1024, num_classes),
			
		)

	def forward(self, input_image):
		N = input_image.shape[0]
		x = self.conv(input_image)
		x = self.model(x)
		x = self.avgpool(x)
		x = x.reshape(N, -1)
		x = self.fc(x)
		return x



class helper_functions(MobileNet):
	
	def train(self,model, train_loader, valid_loader, criterion, optimizer, training_epochs):
		model = model
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
				
				
				
				train_loss += loss.item()
			
			
			
			trainLoss.append(train_loss/len(train_loader))
			trainAcc.append(100.*correct/total)

			valid_loss = 0.0
			valid_acc = 0.0
			total=0.0
			correct=0.0
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
			
			
			min_valid_loss = self.EarlyStopping(model, min_valid_loss, valid_loss/len(valid_loader))

			print(f'Epoch {e+1} Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
			
			
			
		return model,trainLoss, validLoss, trainAcc, validAcc

	def EarlyStopping(self,model ,min_valid_loss, valid_loss):
		min_valid_loss =min_valid_loss
		if min_valid_loss > valid_loss:
			model = model
			#print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
			min_valid_loss = valid_loss
			
			# Saving State Dict
			torch.save(model.state_dict(), 'ErlStpsaved_model.pth') 
		
		return min_valid_loss


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
	def predict(self,model, testX):
		# predict the value of testX
		test_pred=model(testX)
		
		
		return test_pred  #predictions

	def predict_digit(self,model,testX):
		model = model
		test_pred=model(testX)
		_, predicted = test_pred.max(1)
		return predicted

	def accuracy(self, labels,test_pred):
		_, predicted = test_pred.max(1)
		
		total = labels.size(0)
		
		correct = predicted.eq(labels).sum().item()
		
		Acc = (100.*correct/total)	
		
		return Acc # return accuracy    
		
	

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
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


	
# %%
wd_path = os.getcwd()
data_path = 'Data/'


_path=os.path.join(wd_path,data_path)


print("Wd",_path)

print(_path)
batch_size = 128


##################################################################
##   good way to load dataset                   ##
##################################################################

transform = transforms.Compose([
transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

traindataset = myMNISTdata(csv_file='train.csv',root_dir=_path + 'train/train_new/', transform=transform) 
testdataset = myMNISTdata(csv_file='test.csv',root_dir=_path + 'test/test_new/', transform=transform) 
train_set, val_set = torch.utils.data.random_split(traindataset, [36470, 7000])#[3,1])#[36470, 7000])  #total 43470
		
train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(testdataset, 10000, shuffle=False) #making one batch of whole test data
print(traindataset[1][1])
plt.imshow(traindataset[1][0].reshape(28, 28))
plt.show()


# %%

##################################################################
##              Important parameters for the model              ##
##################################################################
plot_err = True
epochs = 10
learning_rate = 0.05

######################################
# Initializing the model 
######################################

MobileNetmodel = MobileNet()
summary(MobileNetmodel, (1, 28, 28))
# the helper_functions class inherits the MobileNet
#So by constructing the helper_function class we initialize the model too
model = helper_functions()

######################################
#Loss function and optimizer
######################################
#criterion = FocalLoss()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(MobileNetmodel.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(MobileNetmodel.parameters(),lr=learning_rate,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)

if torch.cuda.is_available():
		MobileNetmodel.cuda()
		criterion.cuda()

##################################################################
##                        Training The model                    ##
##################################################################
# The EarlyStoping is implemented in the class neural network
#  

MobileNetmodel, trainLoss, validLoss, trainAcc, validAcc=model.train(MobileNetmodel, train_loader, valid_loader, criterion, optimizer, epochs)

#saving the model
torch.save(MobileNetmodel.state_dict(),'models/experimentNo.pth') 


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
helper_func = helper_functions()
trainedModel = MobileNet()
trainedModel.load_state_dict(torch.load('models/experiment4.pth'))
#trainedModel.cpu()
trainedModel.eval()
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
test_pred = helper_func.predict(trainedModel, tdata)
testAcc = helper_func.accuracy( labels,test_pred)
print(torch.unsqueeze(tdata[0], 0).shape)
print(helper_func.predict_digit(trainedModel, torch.unsqueeze(tdata[0], 0)))

################
# Accuracy
################
print("test Accuracy = ",testAcc)

################
# precision Recall and F1-score
################

print(helper_func.precision_recall_f1score(labels, test_pred))


################
#Confusion Matrix calculation
################

helper_func.confusion_Mat(tlabel, test_pred,'Test Data Confusion Matrix')

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
model_weights = [] 
conv_layers = [] 

model_children = list(trainedModel.children())

# counter to keep count of the conv layers
counter = 0 

for i in range(len(model_children)):
	
	if type(model_children[i]) == nn.Conv2d:
		counter += 1
		model_weights.append(model_children[i].weight)
		conv_layers.append(model_children[i])
	elif type(model_children[i]) == nn.Sequential:
		
		for j in range(len(model_children[i])):
			
			for child in model_children[i][j].children():
				
				for ch in child.children():
					for chi in ch.children():
						if type(chi) == nn.Conv2d:
							counter += 1
							model_weights.append(chi.weight)
							conv_layers.append(chi)
							
				
print(f"Total convolutional layers: {counter}")

plt.figure(figsize=(20, 20))
# Printing the convolutional layer so that we can see the filter size and number of filters
for i in range(len(conv_layers)):
	print(conv_layers[i])
# visualizing the final convolution layer filters
# we have 1024 of them i only plotted first 100 of them
for i, filter in enumerate(model_weights[14]):
	plt.subplot(10, 10, i+1) # (3, 3) because in conv14 we have 3x3 filters and total of 1024
	plt.imshow(filter[0, :, :].detach(), cmap='gray')
	plt.axis('off')
	if i == 99:
		break
	plt.savefig('filter.png')
	
plt.show()



# %%
###############################################################
#     tSNE plots
#test data tsne plot 

testdata = tdata.reshape(tdata.shape[0], -1)
data_pred = predicted


helper_func.tSNE(testdata,data_pred,'test data tsne plot')



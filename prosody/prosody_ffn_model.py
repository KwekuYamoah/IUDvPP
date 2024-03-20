import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from tqdm import tqdm 

from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


'''
The MLPNetwork architecture is a deep learning architecture for learning the relationship
between the sequence of amplitude values of the different acoustic characteristics of the speech
of a language and the prosodic prominences and boundaries of that language. 
'''
class MLPNetwork(nn.Module):
    def __init__(self, input_seq_len, output_size, dropout=0.0):
        super(MLPNetwork, self).__init__()
        self.dense_1 = nn.Linear(input_seq_len, output_size)

    

        


    def forward(self, x):
      # Compute prediction
      prediction = self.dense_1(torch.Tensor(x))


      #final prediction
      return prediction
    

'''
The validation_metrics function returns the main validation metrics of precision, recall and accuracy to demonstrate the level of the performance
of the sequence model.

Params:
    model_name (string): This is the name of the model that is being used to generate the predictions.
    model (obj): This is the model object that is being used to generate the predictions.
    test_data (list): This is a list containing the audio data and their corresponding gold labeled vector tags.

Returns:
    results (dict): This is a dictionary containing the validation metrics scorres for accuracy, F1, precision and recall.


Models: LinearModel, SVMModel, RandomForestModel, LogisticRegressionModel, BiLSTMAttentionNetwork, BiLSTMNetwork, RNNAttentionNetwork, 
RNNNetwork, MLPNetwork, TransformerModel, 
'''
def validation_metrics(model, test_data):
    #initialize list to hold the true values and the predicted values
    true_values = []
    pred_values = []

    #iterate through the test data samples and then compute the predictions for each sample point
    for data_sample in test_data:
        #extract the data sequence and its corresponding label from test_data
        x_true = data_sample["input_features"]
        y_true = data_sample["labels"]


        #generate a prediction for the current data sample in the process of the iteration
        y_pred_output = model(x_true)
        
        #figure out which class was predicted for each word
        y_pred = []

        for pred_out in y_pred_output:
            #find the max out of the predicted output for the word
            if isinstance(pred_out, torch.Tensor):
                pred_out = pred_out.tolist()
            
            max_element = max(pred_out)

            predicted_class = pred_out.index(max_element)

            #append the found class to y_pred
            y_pred.append(predicted_class)
        
        #add the predicted values and the true values to their corresponding lists
        true_values += y_true
        pred_values += y_pred
        


    #compute the evaluation metrics
    accuracy = accuracy_score(true_values, pred_values)
    precision = precision_score(true_values, pred_values)
    recall = recall_score(true_values, pred_values)
    f1 = f1_score(true_values, pred_values)




    return {"accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1}

'''
The train_model function trains the sequence model that we have initialized with the training data given.

Params:
    training_data (list): This is a list containing the training data.
    val_data (list): This is a list containing the validation data.
    test_data (list): This is a list containing the testing data.
    learning_rate (float): This is a floating point number that represents the learning rate of the model.
    epochs (int): This is an integer that represents the number of training cycles of the model.


Returns:
    train_f1s (list) : This is a list containing the f1 scores obtained during training.
    val_f1s (list): This is a list containing the f1 scores obtained during model validation.
    test_f1s (list): This is a list containing the f1 scores obtained during model testing.
    train_accuracy (list): This is a list containing the accuracy scores obtained during model training.
    val_accuracy (list): This is a list containing the accuracy scores obtained during model validation.
    test_accuracy (list): This is a list containing the accuracy scores obtained during model testing.
    train_precision (list): This is a list containing the precision scores obtained during model training.
    val_precision (list): This is a list containing the precision scores obtained during model precision. 
    test_precision (list): This is a list containing the precision scores obtained during model testing.
    train_recall (list): This is a list containing the recall scores obtained during model training.
    val_recall (list): This is a list containing the recall scores obtained during model validation.
    test_recall (list): This is a list containing the recall scores obtained during model testing.
    
'''
def train_model(model, training_data, val_data, test_data, learning_rate, epochs, device):

    #initialize the loss function
    criterion = nn.CrossEntropyLoss()

    #initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    #initialize lists to store the validation metrics generated in each epoch
    train_f1s = np.empty(epochs)
    val_f1s = np.empty(epochs)
    test_f1s = np.empty(epochs)
    train_accuracy = np.empty(epochs)
    val_accuracy = np.empty(epochs)
    test_accuracy = np.empty(epochs)
    train_precision = np.empty(epochs)
    val_precision = np.empty(epochs)
    test_precision = np.empty(epochs)
    train_recall = np.empty(epochs)
    val_recall = np.empty(epochs)
    test_recall = np.empty(epochs)




    #iterate through each epoch and compute the corresponding validation metrics
    for epoch in tqdm(range(epochs)):
        #keep track of the running loss during training
        running_loss = 0.0

      
        for input_data in training_data:
            #set the optimizer to zero grad
            optimizer.zero_grad()
            
            #perform a forward propagation
            prediction = model(input_data["input_features"])

            #compute the loss
            loss = criterion(prediction, torch.Tensor(input_data["labels"]))

            #perform backward propagation
            loss.backward()

            #perform optimization step
            optimizer.step()

            #update the running loss
            running_loss += loss.item()
        

        #save the model checkpoint after each epoch of training
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, './epoch_'+str(epoch+1)+'_model_checkpoint.pt')
        
        
        #display the training statistics
        print(f'Epoch: {epoch + 1} loss: {running_loss}')

        # Compute f1-scores on train and val datasets after training for an epoch
        with torch.no_grad():
          train_f1s[epoch] = validation_metrics(model, training_data)['f1']
          val_f1s[epoch] = validation_metrics(model, val_data)['f1']
          test_f1s[epoch] = validation_metrics(model, test_data)['f1']
          train_accuracy[epoch] = validation_metrics(model, training_data)['accuracy']
          val_accuracy[epoch] = validation_metrics(model, val_data)['accuracy']
          test_accuracy[epoch] = validation_metrics(model, test_data)['accuracy']
          train_precision[epoch] = validation_metrics(model, training_data)['precision']
          val_precision[epoch] = validation_metrics(model, val_data)['precision']
          test_precision[epoch] = validation_metrics(model, test_data)['precision']
          train_recall[epoch] = validation_metrics(model, training_data)['recall']
          val_recall[epoch] = validation_metrics(model, val_data)['recall']
          test_recall[epoch] = validation_metrics(model, test_data)['recall']



    return train_f1s, val_f1s, test_f1s, train_accuracy, val_accuracy, test_accuracy, train_precision, val_precision, test_precision, train_recall, val_recall, test_recall







'''
The execute_MLP_model function initializes the MLP network and provides it with the train, test and validation data that it needs to train and 
hone its performance.

Params:
    train (list): This is a list of dictionary values, with each list containing different extracted features from an audio file.
                    This list constitutes the training data for the model.
    test (list): This is a list of dictionary values, with each list containing different extracted features from an audio file.
                    This list constitutes the test data for the model.
    val (list): This is a list of dictionary values, with each list containing different extracted features from an audio file.
                    This list constitutes the validation data for the model.
    input_feature_type (str): This is a string that is used to represent whether the full vector of values of each audio file
                            is fed into the model as a feature vector or whether the extracted statistical features of the
                            full vector of each audio file is fed into the model as a feature vector.

Returns:
    None
'''
def execute_MLP_model(train, test, val, input_size, output_size, dropout, device="cpu"):
    #initialize the mlp model
    model = MLPNetwork(input_size, output_size, dropout)

    #cast the model to the device
    model.to(device)

    #initialize the training, testing and validation of the model
    train_f1s, val_f1s, test_f1s, train_accuracy, val_accuracy, test_accuracy, train_precision, val_precision, test_precision, train_recall, val_recall, test_recall = train_model(model, train, val, test, 0.001, 10, "cpu")

    print("test accuracy: ", test_accuracy)
    print("test precision: ", test_precision)
    print("test recall: ", test_recall)
    print("test_f1: ", test_f1s)




    return 



def generate_prediction(model_path, input_size, output_size, dropout, test_data_input_features):
    #load the model checkpoint
    checkpoint = torch.load(model_path)

    #initialize the model
    model = MLPNetwork(input_size, output_size, dropout)

    #load the model state
    model.load_state_dict(checkpoint['model_state_dict'])

    #generate a prediction
    model_prediction = model(test_data_input_features)

    #initialize list to hold the predicted labels
    predicted_labels = []

    #iterate through the predicted outputs and obtain the predicted class for each word
    for pred_item in model_prediction:
        #check to see if pred_item is a pytorch tensor
        if isinstance(pred_item, torch.Tensor):
            pred_item = pred_item.tolist()
        
        max_value = max(pred_item)

        predicted_class = pred_item.index(max_value)

        predicted_labels.append(predicted_class)

 

    return predicted_labels

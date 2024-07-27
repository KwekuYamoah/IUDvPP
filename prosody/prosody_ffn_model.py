import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import json
import math
import random

from tqdm import tqdm 

from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from torch import Tensor as tensor

# Suppress only UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)



class LSTM_MLP_Network(nn.Module):
    def __init__(self, input_seq_len, lstm_hidden_size, output_size, dropout = 0.0):
        super(LSTM_MLP_Network, self).__init__()
        self.lstm_layer = nn.LSTM(input_seq_len, lstm_hidden_size, batch_first=True, bidirectional=True, num_layers=1)
        self.linear_layer_1 = nn.Linear(lstm_hidden_size*2, lstm_hidden_size*2)
        self.linear_layer_2 = nn.Linear(lstm_hidden_size*2, output_size)
        self.linear_layer_3 = nn.Linear(output_size, 1)
        self.sigmoid_layer = nn.Sigmoid()
    
    def forward(self, x):
        lstm_output, (_, _) = self.lstm_layer(x)
        linear_layer_output_1 = self.linear_layer_1(lstm_output)
        linear_layer_output_2 = self.linear_layer_2(linear_layer_output_1)
        linear_layer_output_3 = self.linear_layer_3(linear_layer_output_2)
        model_prediction = self.sigmoid_layer(linear_layer_output_3)
        
        return model_prediction




  

'''
The validation_metrics function returns the main validation metrics of precision, recall and accuracy to demonstrate the level of the performance
of the sequence model.

Params:
    model_name (string): This is the name of the model that is being used to generate the predictions.
    model (obj): This is the model object that is being used to generate the predictions.
    test_data (list): This is a list containing the audio data and their corresponding gold labeled vector tags.

Returns:
    results (dict): This is a dictionary containing the validation metrics scorres for accuracy, F1, precision and recall.


'''
def validation_metrics(model, test_data):
    model.eval()

    #initialize list to hold the true values and the predicted values
    true_values = []
    pred_values = []

    #iterate through the test data samples and then compute the predictions for each sample point
    for data_sample in test_data:
        #extract the data sequence and its corresponding label from test_data
        x_true = data_sample['input_features']
        y_true = data_sample['labels']


        #generate a prediction for the current data sample in the process of the iteration
        model_prediction = model(x_true)
        model_prediction = model_prediction.squeeze(1)

        y_pred = []

        for i in model_prediction:
            if i >= 0.5:
                y_pred.append(1.0)
            else:
                y_pred.append(0.0)

        
        
        #add the predicted values and the true values to their corresponding lists
        true_values += y_true.tolist()
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


    model.train()

    #iterate through each epoch and compute the corresponding validation metrics
    for epoch in tqdm(range(epochs)):
        #keep track of the running loss during training
        running_loss = 0.0

      
        for input_data in training_data:
            #set the optimizer to zero grad
            optimizer.zero_grad()
            
            #perform a forward propagation
            prediction = model(input_data['input_features'])
            #print('prediction: ', prediction.squeeze(1))
            #print('gold label: ',  input_data['labels'])


            #compute the loss
            loss = criterion(prediction.squeeze(1), input_data['labels'])

            #update the running loss
            running_loss += loss.detach().item()
         

            #perform backward propagation
            loss.backward()

            #perform optimization step
            optimizer.step()

            
        

        #save the model checkpoint after each epoch of training
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, './model_checkpoints/epoch_'+str(epoch+1)+'_model_checkpoint.pt')
        
        
        #display the training statistics
        epoch_value = epoch + 1
        if epoch_value % 100 == 0:
            print(f'Epoch: {epoch_value} loss: {running_loss}')

        # Compute f1-scores on train and val datasets after training for an epoch
        with torch.no_grad():
          #train_f1s[epoch] = validation_metrics(model, training_data, training_data_labels)['f1']
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



def execute_LSTM_MLP_model(train, test, val, input_size, lstm_hidden_size, output_size, dropout, device="cpu"):
    #initialize the mlp model
    model = LSTM_MLP_Network(input_size, lstm_hidden_size, output_size, dropout)

    #cast the model to the device
    model.to(device)

    #initialize the training, testing and validation of the model
    train_f1s, val_f1s, test_f1s, train_accuracy, val_accuracy, test_accuracy, train_precision, val_precision, test_precision, train_recall, val_recall, test_recall = train_model(model, train, val,  test, 0.001, 100, "cpu")

    print("test accuracy: ", test_accuracy[-1])
    print("test precision: ", test_precision[-1])
    print("test recall: ", test_recall[-1])
    print("test f1: ", test_f1s[-1])



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


def split_data(json_data_object, train_proportion, test_proportion, val_proportion):
    #read the data from the input json object
    with open(json_data_object) as inputfile:
        input_data = json.load(inputfile)
    
    #compute the number of elements that should belong to the train, test and val sections
    train_num = math.floor(len(input_data) * train_proportion)
    test_num = math.floor(len(input_data) * test_proportion)
    val_num = math.floor(len(input_data) * val_proportion)

    data_indicies = [i for i in range(len(input_data))]
   

    #randomly select train indicies
    train_indices = []

    for i in range(train_num):
        #generate random number
        train_index = random.randint(0, len(data_indicies)-1)
     

        #append the random generated index to train_indices
        train_indices.append(data_indicies[train_index])
  
    

        #remove the random generated index from data_indices
        del data_indicies[train_index]
      
    

    #construct the train set
    train_set = []

    for train_index in train_indices:
        train_set.append(input_data[train_index])
    
    #randomly select test indicies
    test_indices = []

    for i in range(test_num):
        #generate random number
        test_index = random.randint(0, len(data_indicies)-1)

        #append the random generated index to train_indices
        test_indices.append(data_indicies[test_index])

        #remove the random generated index from data_indices
        del data_indicies[test_index]
    
    #construct the train set
    test_set = []

    for test_index in test_indices:
        test_set.append(input_data[test_index])

    #randomly select train indicies
    val_indices = []

    for i in range(val_num):
        #generate random number
        val_index = random.randint(0, len(data_indicies)-1)

        #append the random generated index to train_indices
        val_indices.append(data_indicies[val_index])

        #remove the random generated index from data_indices
        del data_indicies[val_index]
    

    #construct the train set
    val_set = []

    for val_index in val_indices:
        val_set.append(input_data[val_index])




    return train_set, test_set, val_set


def preprocess_data(input_data):
    #iterate through the data to extract the input features and their corresponding labels
    extracted_features = []
    features_lengths = []

    extracted_labels = []
   

 


    for entry in input_data:
        filename = list(entry.keys())[0]
        feature_list = entry[filename]["input_features"]
        label_list = entry[filename]["labels"]

        for feature_vector in feature_list:
            round_feature_vector = [round(i, 2) for i in feature_vector]
            extracted_features.append(round_feature_vector)
            features_lengths.append(len(feature_vector))
        
        
        extracted_labels += label_list
        
    

    #get the max for the feature lengths
    max_feature_lengths = max(features_lengths)
  


    processed_feature_vectors = []
   


    #ensure that all of the extracted features and their corresponding labels have the same lengths
    for feature_vector in extracted_features:
        if len(feature_vector) < max_feature_lengths:
            #find out the length difference
            len_diff = max_feature_lengths - len(feature_vector)

            #generate a list of zeros and append it to the feature vector
            zeros_vector = [0] * len_diff

            processed_feature_vector = feature_vector + zeros_vector

            processed_feature_vectors.append(processed_feature_vector)
        
        else:
            processed_feature_vectors.append(feature_vector)
    

   
    


    return processed_feature_vectors, extracted_labels





def count_data_label_values(data_list):
    one_count = 0
    zero_count = 0

    for data_value in data_list:
    
        if data_value == 1.0:
            one_count += 1
        else:
            if data_value == 0.0:
                zero_count += 1
    
    print('ones: ', one_count)
    print('zeros: ', zero_count)

    
    return


def balance_train_dataset(train_features, train_labels):
    #get the features with a label of 1 first
    label_one_input_features = []
    label_one_labels = []

    label_zero_input_features = []
    label_zero_labels = []

    balanced_zero_input_features = []
    balanced_zero_labels = []
    generated_zero_indicies = []

    for data_index in range(len(train_features)):
        if train_labels[data_index] == 1.0:
            label_one_input_features.append(train_features[data_index])
            label_one_labels.append(train_labels[data_index])
        
        else:
            label_zero_input_features.append(train_features[data_index])
            label_zero_labels.append(train_labels[data_index])
    

    
    

    #randomly select items from the zero label list to ensure that the number of items in the one list is the same as the number of items
    #in the zero list
    while len(balanced_zero_input_features) < len(label_one_input_features):
        #generate a ranom number
        random_index = random.randint(0, len(label_zero_input_features)-1)

        if random_index not in generated_zero_indicies:
            balanced_zero_input_features.append(label_zero_input_features[random_index])
            balanced_zero_labels.append(label_zero_labels[random_index])

            generated_zero_indicies.append(random_index)
    





    #join the two different input features together
   
    balanced_dataset = label_one_input_features + balanced_zero_input_features
    balanced_labels = label_one_labels + balanced_zero_labels

    

    return balanced_dataset, balanced_labels



def find_data_vector_max_length(data_json_list, vector_type):
    data_vectors_lengths = []

    #iterate through data json
    if vector_type == 'labels':
        for data_group in data_json_list:
            for data_entry in data_group:
                #get the file name of the current data entry
                filename = list(data_entry.keys())[0]

                #get the input feature vector of the current data entry
                feature_list = data_entry[filename][vector_type]

                #append the len of the feature vector to the data_vectors_lengths list
                data_vectors_lengths.append(len(feature_list))
    else:
        for data_group in data_json_list:
            for data_entry in data_group:
                #get the filename of the current data entry
                filename = list(data_entry.keys())[0]

                #get the input feature vector of the current data entry
                feature_list = data_entry[filename][vector_type]

                #intialize varibale to count the number of rows in the feature list
                row_count = 0

                #iterate through the feature list to count the number of rows
                for feature_vector in feature_list:
                    row_count += 1
                
                data_vectors_lengths.append(row_count)


        

    #return the max length in the data_vectors_lengths list
    return max(data_vectors_lengths)



def standardize_dataset(dataset_json_list):
    returned_dataset_json_list = []


    #get the max length of the label vectors
    max_length_label_vectors = find_data_vector_max_length(dataset_json_list, 'labels')
    max_length_input_feature_matrix_rows = find_data_vector_max_length(dataset_json_list, 'input_features')
 



    #iterate through the feature json information and ensure that the input features and their corresponding labels have the same corresponding lengths
    for data_obj in dataset_json_list:
        standardized_dataset = []

        for data_entry in data_obj:
            #get the file name of the current data entry
            filename = list(data_entry.keys())[0]

            #get the label vector and ensure that it also meets the max length requirement for label vectors and if not, pad it
            label_list = data_entry[filename]['labels']
            input_features = data_entry[filename]['input_features']
            
            if len(label_list) != 0 and len(input_features) != 0:
                if len(label_list) < max_length_label_vectors:
                    label_diff_len = max_length_label_vectors - len(label_list)
                    label_pad_list = [0] * label_diff_len
                    label_list = label_list + label_pad_list
                

                #round the entries in the input features to 2dp
                round_input_features = []

                for input_feature_vector in input_features:
                    round_vector = [round(i,2) for i in input_feature_vector]
                    round_input_features.append(round_vector)
                
                #make round_input_features a tensor object
                input_feature_matrix = tensor(round_input_features)

                #pad the input feature matrix to ensure that all of the input feature matrices have the same dimensions
                input_feature_matrix = F.pad(input_feature_matrix, (0, 0, 0, max_length_input_feature_matrix_rows - input_feature_matrix.size(0)))
                
                #append the label list and its corresponding round input features to the standardized dataset list
                standardized_dataset.append({'input_features': input_feature_matrix, 'labels': tensor(label_list)})
            
            returned_dataset_json_list.append(standardized_dataset)

        
    return returned_dataset_json_list



def batch_data(input_data, batch_size):
    #initialize variable to hold the data batches
    batched_data = []

    initial_batch_index = 0
    terminal_batch_index = batch_size

    while terminal_batch_index < len(input_data):
        batched_data.append(tensor(input_data[initial_batch_index : terminal_batch_index]))

        initial_batch_index = terminal_batch_index
        terminal_batch_index += batch_size
    
    if terminal_batch_index >= len(input_data):
        batched_data.append(tensor(input_data[initial_batch_index : ]))
    



    return batched_data

if __name__ == '__main__':
    #obtain the train,test and val datasets
    train, test, val = split_data('./en_extracted_features.json', 0.80, 0.10, 0.10)

    #further process the data to ensure that both the input features and the labels have the same length
    preprocessed_data = standardize_dataset([train, test, val])

    train_data = preprocessed_data[0]
    test_data = preprocessed_data[1]
    val_data = preprocessed_data[2]


    '''
    processed_train, train_labels = preprocess_data(train)
    balanced_train, balanced_labels = balance_train_dataset(processed_train, train_labels)
    batched_balanced_train = batch_data(balanced_train, 10)
    batched_balanced_train_labels = batch_data(balanced_labels, 10)
    processed_test, test_labels = preprocess_data(test)
    processed_val, val_labels = preprocess_data(val)
    '''


    #execute the mlp model
    #execute_MLP_model(train_data, test_data, val_data, 7, 1, 0.0, device="cpu")
    execute_LSTM_MLP_model(train_data, test_data, val_data, 7, 7, 54, 0.0, device="cpu")
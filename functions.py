#TODO: Point adjusted F1 hinzufügen
import pandas as pd
import numpy as np
import sklearn
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import constants


''' functions for evaluation calculations'''

def get_f1(validation_labels, inference_labels):

    from sklearn.metrics import f1_score

    f1 = f1_score(validation_labels, inference_labels)

    return f1



def get_ROC_AUC(validation_labels, inference_labels):
    
    from sklearn.metrics import roc_curve, roc_auc_score
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(validation_labels, inference_labels)
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(validation_labels, inference_labels)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    return roc_auc


def get_precision(validation_labels, inference_labels):
    from sklearn.metrics import precision_score
    precision = precision_score(validation_labels, inference_labels)
    return precision

def get_recall(validation_labels, inference_labels):
    from sklearn.metrics import recall_score
    recall = recall_score(validation_labels, inference_labels)
    return recall

def get_accuracy(validation_labels, inference_labels):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(validation_labels, inference_labels)
    return accuracy


def get_recall_precision_curve(validation_labels, inference_scores):
    from sklearn.metrics import precision_recall_curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(validation_labels, inference_scores)
    # plot the model precision-recall curve
    plt.plot(recall, precision, marker='.', label='Test')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall-Precision Curve')
    plt.grid(True)
    plt.show()
    

def get_confusion_matrix(validation_labels, inference_labels):

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    # Create confusion matrix
    cm = confusion_matrix(validation_labels, inference_labels)

    # Create labels for the matrix
    labels = ['Normal', 'Anomaly']

    # Create a DataFrame from the confusion matrix
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    print(cm_df)

    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()



def get_segmentwise_f1(df):
    
    #df has the columns labels and predictions, 
    
    return None



''' functions for model prediction'''
    
def predict_AE(model, data, sequence_length):
    """
    Predict anomalies using an Autoencoder model.
    
    Args:
        model (object): Autoencoder model used for prediction.
        data (numpy.ndarray): Input data to be predicted.
        Removed and commented out: threshold (float): Threshold value for classifying anomalies.
        
    Returns:
        pandas.DataFrame: DataFrame containing anomaly scores for each data point.
    """
    
    
    X = data
    
    data = transform_data(data, sequence_length)
    
    # Let the model create a representation of the original input using the Autoencoder
    data_pred = model.predict(data)
    
    error = abs(data - data_pred)
    
    score = []
    
    for i in error:
        score.append(i.mean())
        
    scored = pd.DataFrame(index=X.index[:-sequence_length])
    
    scored['Loss_mse'] = np.abs(score)    
    
    return scored

    
    
def transform_data(data, sequence_length):
    
    '''
    Create scaled values and time shift
    '''
    
    data_length = len(data)  # Calculate the length of the input data

    new = []  # List to create data set
    
    for i in range(0, data_length - sequence_length):
        
        new.append(data[i:i+sequence_length])  # Add input sequence to Train
        
    new = np.array(new)
        
    # Return the generated datasets and related information
    return new
    



    
def get_embeddings(model, data,sequence_length, mode='TSNE'):
    """
    Get the embedding vectors of each observation using a Keras Sequential model.

    Args:
        model (object): Sequential model used for getting embeddings.
        data (numpy.ndarray): Input data for which embeddings are to be obtained.
        mode (str, optional): Dimensionality reduction mode. Defaults to 'TSNE'.
        
    Returns:
        numpy.ndarray: Embedding vectors of each observation.
    """
    encoder_model = Model(inputs=model.input, outputs=model.get_layer('bottleneck').output)

    # Encode the data using the encoder model
    encoded_data = encoder_model.predict(transform_data(data, sequence_length))

    return encoded_data




def select_features(self, feature_selection):
    
    feature_dict = {}

    if feature_selection:
        # Calculate SOLL-IST differences for voltage measurements
        soll_Spannung = (self.df_raw['Spannung_L1_L2_Hauptantrieb'] + self.df_raw['Spannung_L2_L3_Hauptantrieb'] + self.df_raw['Spannung_L3_L1_Hauptantrieb']) / 3
        self.df_raw['Spannung_L1_L2_abweichung_mittlere_Leitungsspannung'] = soll_Spannung - self.df_raw['Spannung_L1_L2_Hauptantrieb']
        self.df_raw['Spannung_L2_L3_abweichung_mittlere_Leitungsspannung'] = soll_Spannung - self.df_raw['Spannung_L2_L3_Hauptantrieb']
        self.df_raw['Spannung_L3_L1_abweichung_mittlere_Leitungsspannung'] = soll_Spannung - self.df_raw['Spannung_L3_L1_Hauptantrieb']

# Calculate SOLL-IST differences for current measurements
        soll_Strom = (self.df_raw['Strom_L1_Hauptantrieb'] + self.df_raw['Strom_L2_Hauptantrieb'] + self.df_raw['Strom_L3_Hauptantrieb']) / 3
        self.df_raw['Strom_L1_abweichung_mittlerer_Leitungsstrom'] = soll_Strom - self.df_raw['Strom_L1_Hauptantrieb']
        self.df_raw['Strom_L2_abweichung_mittlerer_Leitungsstrom'] = soll_Strom - self.df_raw['Strom_L2_Hauptantrieb']
        self.df_raw['Strom_L3_abweichung_mittlerer_Leitungsstrom'] = soll_Strom - self.df_raw['Strom_L3_Hauptantrieb']

        # Initialize dictionary keys for features
        for feature in constants.MANUAL_FEATURES:
            feature_dict[feature] = []

    if not feature_selection:
        # Select columns with numerical data types (int and float)
        numerical_columns = self.df_raw.select_dtypes(include=['number']).columns

        # Create a new DataFrame with only the numerical columns
        df_numerical = self.df_raw[numerical_columns]

        # Initialize dictionary keys for features
        for feature in df_numerical.columns:
            feature_dict[feature] = []

    return feature_dict


def build_autoencoder(self, num_features, sequence_length, embedding_layer_size):
	"""
	Build and compile the autoencoder model.

	Args:
		num_features (int): Number of features in the input data.
		sequence_length (int): The length of input sequences for the autoencoder.
		embedding_layer_size (int): The size of the embedding layer in the autoencoder.

	Returns:
		keras.models.Model: Compiled autoencoder model.
	"""
	model = keras.Sequential()
	model.add(LSTM(64, batch_input_shape=(None, sequence_length, num_features), return_sequences=True, name='encoder_1', activation='relu'))
	model.add(LSTM(int(embedding_layer_size), return_sequences=False, name='bottleneck'))
	model.add(RepeatVector(sequence_length, name='encoder_decoder_bridge'))
	model.add(LSTM(int(embedding_layer_size), return_sequences=True, name='decoder_1'))
	model.add(LSTM(64, return_sequences=True, name='decoder_2', activation='relu'))
	model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_features)))

	# Compile the model with Mean Absolute Error (MAE) loss and the Adam optimizer
	model.compile(loss="mae", optimizer='adam')

	# Build the model
	model.build()

	return model







import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import time
import warnings
import sklearn
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis

import keras
from keras.models import load_model
import joblib

import constants
import functions 
from functions import *



class AnomalyDetection:
    '''
    This Class is a Superclass to the Training and Inference class and inherets the initialization and preprocessing which should be similar to those classes.
    '''
    def __init__(self, df):
        """
        Load a DataFrame that will be used for Training and delete useless rows (any that don't contain values for the voltage or current)

        Args:
        df (pd.DataFrame) := The input DataFrame containing voltage and current measurements.
        
        TODO:
        ADD:
        training_start_date :=
        validation_start_date :=
        test_start_date :=
        REPLACE AND OPTIMIZE:
        constants.TRAINING_SPLIT
        constants.VALDIDATION_SPLIT
        constants.TEST_SPLIT

        Output:
        None
        """
        
        #Initialize feature selection variable for model naming
        feature_selection = None
        
        # We only consider data, where there is information about the Spannung or Strom values.
        self.df_raw = df.dropna()
        
        #Now since we deleted data which doesn't contain interesting fields we reindex the whole Dataframe, so that it starts at index 0 again
        self.df_raw.reset_index(inplace=True, drop=True)
        
        

        print("Raw Data imported")
       
    
    def preprocessing(self, feature_selection = True):
        """
        Perform preprocessing on the input DataFrame to prepare it for training.
        This method extracts relevant features, calculates the SOLL-IST differences for voltage and current measurements,
        computes statistical measures for each feature in overlapping time windows, and creates a DataFrame for further 
        analysis and training.

        Args:
        None

        Returns:
        pd.DataFrame: A DataFrame containing preprocessed aggregated data with statistical measures for each feature.
        
        Req:
        DataFrame needs to be in a format where the datetime column is still a column and not still an index
        """

        start = time.time()
        
        # Save feature selection
        self.feature_selection = feature_selection
        
        # Create an empty dictionary to store feature data (manual_features is either True or False)
        feature_dict = functions.select_features(self, feature_selection)
        
        # Create a DataFrame to store coefficients
        self.X = pd.DataFrame()

        # Process each feature
        for feature in feature_dict:
            
            print(feature)
            
            # Create a temporary list to store autocorrelation coefficients
            temporary_list = []

            # Create overlapping windows for feature processing
            for window in range(0, self.df_raw.shape[0] - constants.WINDOW_SIZE_PREPROCESSING, constants.STEP_SIZE_PREPROCESSING):
                # Extract data for the current window
                temp = self.df_raw[feature].values[window: window + constants.WINDOW_SIZE_PREPROCESSING]

                # Append data to the feature dictionary
                feature_dict[feature].append(temp)

            # Calculate statistical measures and add to the DataFrame
            self.X[feature + '_mean'] = pd.Series(feature_dict[feature]).apply(lambda x: x.mean())
            self.X[feature + '_std'] = pd.Series(feature_dict[feature]).apply(lambda x: x.std())
        
        # Drop rows with missing values and reset the index (Drop nan values not only subset)
        self.X.dropna(inplace=True)
        self.X.reset_index(inplace=True, drop=True)

        # Create a list to store timestamps
        time_list = []
        
        #TODO: Die untere Loop kann wahrscheinlich oben in die loop reingemacht werden kann man sich überlegen um code zu verschönern
        # Assign timestamps to each window (at the end! wenn am ende, dann heißt das die batch zum time stamp liegt in vergangenheit --> kein future sight happening)
        for i in range(0, self.df_raw.shape[0] - constants.WINDOW_SIZE_PREPROCESSING, constants.STEP_SIZE_PREPROCESSING):
            # Save the first timestamp for each window
            time_list.append(self.df_raw['time'][i+constants.WINDOW_SIZE_PREPROCESSING])

        # Append timestamps to the DataFrame
        self.X['time'] = pd.Series(time_list)
        self.X['time'] = pd.to_datetime(self.X['time'])
        self.X.set_index('time', inplace=True)
        
        self.df_raw["time"] = pd.to_datetime(self.df_raw["time"])
        self.df_raw.set_index("time", inplace = True)

        return self.X

    
    
    
    
    
    

    
class Training(AnomalyDetection):
    '''
    Is a child of the Class AnomalyDetection and inherets the init and preprocessing from it
    '''
    def __init__(self, df):
        super().__init__(df)
        
    def trainUnsupervised(self, saving_path, epochs, embedding_layer_size=8, sequence_length=10):
        """
        Train an unsupervised LSTM autoencoder model for anomaly detection.

        This method preprocesses the input data, scales it, and trains a Long Short-Term Memory (LSTM) autoencoder model
        for anomaly detection. The trained model is saved along with the scaler used for preprocessing.

        Args:
            saving_path (str): The path where the trained model and scaler will be saved.
            epochs (int): The number of training epochs for the LSTM autoencoder.
            embedding_layer_size (int): The size of the embedding layer in the autoencoder. Default is 8.
            sequence_length (int): The length of input sequences for the autoencoder. Default is 10.

        Returns:
            None

        """
        self.sequence_length = sequence_length
        
        print(str(self.feature_selection))
        start = time.time()

        # Initialize a StandardScaler for data normalization
        scaler = StandardScaler()

        # Split the preprocessed data into training, validation, and test sets
        X_train = self.X.loc[self.X.index < constants.VALIDATION_SPLIT]
        X_vali = self.X.loc[(self.X.index > constants.VALIDATION_SPLIT) & (self.X.index < constants.TEST_SPLIT)]
        X_test = self.X.loc[(self.X.index > constants.TEST_SPLIT)]

        # Fit the scaler on the training data
        scaler.fit(X_train)
        
        # TODO: Funktionen aufbrechen und in functions auslagern falss gewünscht für Modulareren Code
        # Transform both training and validation data using the same scaler
        X_train_scaled = scaler.transform(X_train)
        X_vali_scaled = scaler.transform(X_vali)
        X_test_scaled = scaler.transform(X_test)

        # Transform the data into sequences for input to the autoencoder
        X_train_transformed = functions.transform_data(X_train_scaled, self.sequence_length)
        X_vali_transformed = functions.transform_data(X_vali_scaled, self.sequence_length)
        X_test_transformed = functions.transform_data(X_test_scaled, self.sequence_length)

        # Print information about the data splits
        print("Training Data: " + str(constants.TRAIN_SPLIT) + " until " + str(constants.VALIDATION_SPLIT))
        print("Validation Data: " + str(constants.VALIDATION_SPLIT) + " until " + str(constants.TEST_SPLIT))
        print("Validation Data: " + str(constants.TEST_SPLIT) + " until latest")

        # Define the number of features in the input data
        num_features = len(self.X.columns)
        
        #Build the autoencoder model in a seperate function
        model = functions.build_autoencoder(self, num_features, self.sequence_length, embedding_layer_size)

        # Assuming you have a validation set, adjust this according to your data
        validation_data = (X_vali_transformed, X_vali_transformed)

        # Train the autoencoder model and store training history
        history = model.fit(X_train_transformed, X_train_transformed, epochs=epochs, validation_data=validation_data)

        # Save the trained model
        model_filename = str(saving_path) + "model_LSTM_AE_" + "embedding_" + str(embedding_layer_size) + "_epochs_" + str(epochs) + "_sequence_length_" + str(self.sequence_length) + "_unsupervised_" + str(datetime.datetime.now().strftime("%Y%m%d") + "feature_selection_" + str(self.feature_selection))
        model.save(model_filename)

        # Save the scaler used for preprocessing
        scaler_filename = model_filename + "/scaler"
        joblib.dump(scaler, scaler_filename)

        end = time.time()

        print('time: ' + str(end-start))

        # Plot and save the loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(model_filename + "/Loss_plt")
    
    
    
    
    
    
    
    
    
class Inference(AnomalyDetection):
    '''
    Is a child of the Class AnomalyDetection and inherets the init and preprocessing from it
    '''
    def __init__(self, df):
        super().__init__(df)
    
    
    
    def predictAnomalyScores(self, path , model_name, sequence_length = 3):
        """
        Predict anomalies using a pretrained Autoencoder model.

        Args:
            model_name (str): Name of the Autoencoder model to be used for prediction.
            
            path (str): Path where the model and the additional scaler etc. lies

        Returns:
            pandas.DataFrame: DataFrame containing anomaly scores for each data point.
        """                
        start = time.time()

        self.sequence_length = sequence_length
        
        # Load a pretrained Autoencoder model.
        self.model_LSTM_AE = load_model(path + '/' + model_name)
        model_LSTM_AE = self.model_LSTM_AE

        # Load the scaler associated with the model (Each model has its own scaler).
        scaler_filename = path + '/' + model_name + '/scaler'
        loaded_scaler = joblib.load(scaler_filename)

        # Transform the input data using the loaded scaler.
        X_scaled = loaded_scaler.transform(self.X)

        # Create a DataFrame from the scaled data with the same index as the original data.
        X_scaled_df = pd.DataFrame(data=X_scaled, index=self.X.index)

        # Use the Autoencoder model to predict anomalies on the scaled data.
        self.results = predict_AE(model_LSTM_AE, X_scaled_df, self.sequence_length) 

        end = time.time()
        print(end - start)

        # Return the DataFrame containing anomaly scores.
        return self.results

    
  
    def assignHealthStatus(self, threshold = 0.4):
        """
        Pass a threshold to the Autoencoder, to decide which Anomalyscoring is 'too much'. Values above that thresholds are
        considered Anomalys. Adds a 'Label' column to the results DataFrame

        Args:
            threshold (float): Threshold, that is the barrier between normal and not normal data scorings

        Returns:
            None
            
        TODO: Automate the thresholding if wanted
        """  
        
        #Define the threshold
        condition = self.results['Loss_mse'] > threshold
        
        #Add new Column with Labels
        self.results['Label_LSTM_AE'] = np.where(condition, 1, 0)
        
        
        
    def explain_anomalys(self):

        """
        Explain anomalies detected by the model.

        This function explains anomalies detected by the model by analyzing the reconstruction errors, identifying the top
        contributing features for each anomaly, and returning a DataFrame containing explanations for the detected anomalies.

        Returns:
            pandas.DataFrame: A DataFrame containing explanations for the detected anomalies.

        Note:
            This function assumes that the model has already been trained and that anomaly detection results are available
            in the 'results' attribute of the class instance.
        """

        model_LSTM_AE = self.model_LSTM_AE
        y = transform_data(self.X, sequence_length=self.sequence_length)
        y_pred = model_LSTM_AE.predict(transform_data(self.X, sequence_length=self.sequence_length))

        mae = abs(y - y_pred)
        reconstruction_errors = mae.mean(axis=1)

        reconstruction_error_df = pd.DataFrame(
            data=reconstruction_errors, index=self.X.index[:-int(self.sequence_length)], columns=self.X.columns
        )

        reconstruction_error_df['mae_sum'] = reconstruction_error_df.sum(axis=1)

        relative_df = reconstruction_error_df.div(reconstruction_error_df['mae_sum'], axis=0)

        del relative_df['mae_sum']

        def find_max_columns(row):
            sorted_columns = row.sort_values(ascending=False)
            return (sorted_columns.index[0], sorted_columns.index[1], sorted_columns.index[2])

        max_columns = relative_df.apply(find_max_columns, axis=1)
        self.results['reason 1'] = [col[0] for col in max_columns]
        self.results['reason 2'] = [col[1] for col in max_columns]
        self.results['reason 3'] = [col[2] for col in max_columns]

        explained_anomalys = self.results.loc[self.results['Label_LSTM_AE'] == 1]

        return explained_anomalys

    
    
    def loadValidationData(self):
        """
        Load and preprocess validation data from a CSV file.

        Loads the breakdowns.

        Returns:
            pandas.DataFrame: A DataFrame containing the preprocessed validation data.
        """

        #Load Anomalys from the Schichtbuch
        self.validation_df = pd.read_csv('Störungen_data/validation_df')

        self.validation_df['DATUM'] = pd.to_datetime(self.validation_df['DATUM'], errors='coerce')

        self.validation_df['DATUM'].dropna(inplace=True)

        self.validation_df = self.validation_df[(pd.notnull(self.validation_df['DATUM']))]

        self.validation_df['time'] = pd.to_datetime(self.validation_df['DATUM'].apply(lambda x: str(x)[:-8]) +self.validation_df['ZEIT'].astype(str))

        self.validation_df.DAUER = pd.to_numeric(self.validation_df.DAUER, errors='coerce')

        self.validation_df.set_index('time', inplace=True)

        self.validation_df = self.validation_df.sort_index()

        self.validation_df = self.validation_df.loc[(self.validation_df['Stufe3'].str.contains('Haupt') == True) & (self.validation_df['Stufe2'].str.contains('Zentrale') == True)]
        
        #Load Anomalys from the manual Labelings
        self.manual_labels = pd.read_csv("Störungen_data/manual_labels")


    
    def evaluate(self, leeway = 8):
        
        """
        Evaluate the model's performance based on anomaly detection.

        This function evaluates the model's performance by comparing its anomaly predictions with ground truth labels,
        calculating various metrics, and displaying the results.

        Args:
            leeway (int): The time window in hours to consider when evaluating anomalies around each data point.

        Returns:
            dict: A dictionary containing performance metrics, including F1-Score, Confusion Matrix, ROC-AUC, Precision, Recall,
                  Accuracy, and Precision-Recall Curve.

        Note:
            If no anomalies are found, it prints 'No Anomalys found' and returns None.
        """
        #Load Dataframe that contains the Schichtbuch data
        df_störungen = self.validation_df
        
        #Create list to save values to
        störungsbereiche = []
        
        #Use the Manual Labels and add them to the List of timestamps that resemble Anomalys
        for index, row in self.manual_labels.iterrows():
            
            start = row["start"]
            end = row["end"]
            
            subset = self.results.loc[start: end]
            
            störungsbereiche.append(subset.index)
        
        #Mark the anomalous data
        temporary_list = np.array(df_störungen.index)
        for störung in temporary_list:
            
            start = str(pd.to_datetime(störung, utc = True) - pd.Timedelta(hours=leeway))
            end = str(pd.to_datetime(störung, utc = True))
            
            subset = self.results.loc[start: end]
            
            störungsbereiche.append(subset.index)
    
        #Append a new column named 'störung', which contains the value 0 if there is no anomaly and 1 if there is
        for j in range(len(störungsbereiche)):
            self.results.loc[self.results.index.isin(störungsbereiche[j]), 'störung'] = 1
        self.results['störung'].fillna(0, inplace=True)
        
        if(len(self.results.loc[self.results['Label_LSTM_AE'] == 1]) > 0):
            print('F1-Score: ')
            print(get_f1(self.results['störung'], self.results['Label_LSTM_AE']))
            print('Confusion matrix: ')
            print(get_confusion_matrix(self.results['störung'], self.results['Label_LSTM_AE']))
            print('ROC-AUC: ')
            print(get_ROC_AUC(validation_labels = self.results.störung, inference_labels = self.results.Loss_mse))
            print('Precision: ')
            print(get_precision(validation_labels = self.results.störung, inference_labels = self.results.Label_LSTM_AE))
            print('Recall: ')
            print(get_recall(validation_labels = self.results.störung, inference_labels = self.results.Label_LSTM_AE))
            print('Accuracy: ')
            print(get_accuracy(validation_labels = self.results.störung, inference_labels = self.results.Label_LSTM_AE))
            print('Precision-Recall-Curve: ')
            print(get_recall_precision_curve(validation_labels = self.results.störung, inference_scores = self.results.Loss_mse))
            
        else: 
            print('No Anomalys found')

        
    
    
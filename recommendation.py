"""Do recommendataion task.

Author: DHSong
Last Modified At: 2020.07.12

Anticipate target value. Find wheter the user would listen to the song again in a month. 
"""

import os

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.decomposition import NMF
from tensorflow import keras
from tqdm import tqdm

from neural_net import NeuralNetModel

class RecommendationWorker:
    """Worker for recommendation.

    Do recommendation task based on WSDM data. 

    Attributes:
        train (DataFrame): Preprocessed train dataset.  
        validation (DataFrame): Preprocessed validation dataset, splitted from processed train dataset.
        test (DataFrame): Preprocessed test dataset.
        model_nn (NeuralNetModel): Neural Net Model for recommendation.
        model_nmf (NMF): NMF Model for recommendation.
    """    

    def __init__(self, data_dirname='./data', font_path='./static/fonts/D2Coding.ttc'):
        """Initialize data and models.
        
        Args:
            data_dirname (str): Directory path for datasets. 
            font_path (str): File path for fonts, which supports korean language.
        """

        self._matplotlib_setting(font_path)

        self.train = pd.read_csv(os.path.join(data_dirname, 'train_merged.csv'), dtype={'registered_via':'str'})
        self.test = pd.read_csv(os.path.join(data_dirname, 'test_merged.csv'), dtype={'registered_via':'str'})

        self.train, self.validation = self._split_train_validation(size=0.2)

        self.model_nn = NeuralNetModel()
        self.model_nmf = NMF(
            n_components=500, 
            init='random', 
            tol=5e-2, 
            max_iter=50, 
            random_state=200, 
            verbose=1, 
            shuffle=True
        )

    def _matplotlib_setting(self, font_path):
        """Initialize maplotlib settings."""

        font_family = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.size'] = 14

        plt.style.use('seaborn-darkgrid')

    def _split_train_validation(self, size):
        """Split train dataset into train dataset and validation dataset.
        
        Args:
            size (float): 0 to 1. Proportion of test datset. 
        Return:
            train (DataFrame): Splitted train dataset.
            validation (DataFrame): Splitted validation dataset.
        """

        train, validation = train_test_split(
            self.train, 
            test_size=size, 
            random_state=2020, 
            shuffle=True, 
            stratify=self.train.target
        )
        return train, validation

    def _one_hot_encoder(self, dataframe):
        """Get one-hot encoder.
        
        Return one-hot encoder fitted by dataframe.

        Args:
            dataframe (DataFrame): Datasets to be fitted.
        Return:
            encoder (OneHotEncoder): Trained one-hot encoder.
        """

        ### features to be one-hot encoded in dataframe
        features = [
            'source_system_tab', 'source_screen_name', 'source_type',
            'city', 'gender', 'registered_via', 'membership_dyas_bin', 
            'bd_bin', 'language', 'song_length_bin', 'country', 'year_bin'
        ]

        encoder = OneHotEncoder()
        encoder.fit(dataframe[features])

        ### what values in features are transformed to.
        print('*****Encoding Categories*****')
        for feature, categories in zip(features, encoder.categories_):
            print('{}\t{}'.format(feature, categories))

        return encoder

    def _draw_history(self, history, full=False):
        """Plot neural network model training history.

        Draw plot for nueral network model training history. 

        Attributes:
            history: neural netowrk model training history.
        """    

        ### Draw AUC history
        plt.figure(figsize=(16, 9))        
        plt.plot(history.history['auc'], label='train')
        plt.plot(history.history['val_auc'], label='validation')
        plt.title('model AUC')
        plt.xlabel('epoch')
        plt.ylabel('AUC')
        plt.legend(loc='upper right')
        plt.savefig('./figures/{}-model-auc'.format('full' if full else 'partial'))

        plt.clf()

        ### Draw loss history
        plt.figure(figsize=(16, 9))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
        plt.savefig('./figures/{}-model-loss'.format('full' if full else 'partial'))

    def plot_roc_curves(self, real, predictions, scores, names):
        """Compare models.

        Compare models using ROC curves. 

        Args:
            real (ndarray): target data.
            predictions (list): list of predictions.
            scores (list): list of auc scores.
            names (list): list of name of models.
        """    

        plt.figure(figsize=(16, 9))

        for idx in range(len(predictions)):
            prediction = predictions[idx]
            score = scores[idx]
            name = names[idx]

            fpr, tpr, _ = roc_curve(real, prediction)
            plt.plot(fpr, tpr, marker='x', label='{}-{:.3f}'.format(name, score))
        
        plt.title('ROC Curve Score: {:.4f}'.format(score))
        plt.xlabel('False Posivie Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='upper right')
        plt.savefig('./figures/roc_curves')

    def load_dataset_for_nn(self):
        """Load necessary datasets for training neural net.
        
        Get train, validation, and test datasets for training neural network model.
        
        Args:
        Return:
            x_train (ndarray): feature data in train dataset.
            y_train (ndarray): target data in train dataset.
            x_validation (ndarray): feature data in validation dataset.
            y_validation (ndarray): target data in validation dataset.
            x_test (ndarray): feature data in test dataset.
        """

        ### features to be one-hot encoded
        features = [
            'source_system_tab', 'source_screen_name', 'source_type',
            'city', 'gender', 'registered_via', 'membership_dyas_bin', 
            'bd_bin', 'language', 'song_length_bin', 'country', 'year_bin'
        ]

        ### genre featres, which is already one-hot encoded
        genre_features = [
            'genre_465', 'genre_958', 'genre_1609', 'genre_2022', 'genre_2122'
        ]

        y_train = self.train[['target']].to_numpy()
        y_validation = self.validation[['target']].to_numpy()

        x_train = self.train[features]
        x_validation = self.validation[features]
        x_test = self.test[features]

        x_train_validation = pd.concat([x_train, x_validation], axis=0)
        encoder = self._one_hot_encoder(x_train_validation)

        x_train = encoder.transform(x_train)
        x_train = np.hstack((x_train.todense(), self.train[genre_features].to_numpy()))

        x_validation = encoder.transform(x_validation)
        x_validation = np.hstack((x_validation.todense(), self.validation[genre_features].to_numpy()))

        x_test = encoder.transform(x_test)
        x_test = np.hstack((x_test.todense(), self.test[genre_features].to_numpy()))

        return (x_train, y_train), (x_validation, y_validation), x_test   

    def load_dataset_for_nmf(self):
        """Load necessary datasets for training matrix factorization model.
        
        Get user-item pairs, dictionaries, and user-item matrix for training.
        
        Args:
        Return:
            user_item_train (list): user-item pair in train dataset.
            user_item_validation (list): user-item pair in validation dataset.
            user_item_test (list): user-item pair in test dataset.
            user2idx (dict): user to index dictionary
            song2idx (dict): song to index dictionary
            train_matrix (dict): matrix to be factorized. made from train dataset.
        """

        ### features for user item pair
        features = [
            'msno', 'song_id'
        ]

        ### get all user and item values in datasets.
        dataset = pd.concat(
            [self.train[features], self.validation[features], self.test[features]], axis=0
        )
        user2idx = {user:idx for idx, user in enumerate(dataset.msno.unique())}
        item2idx = {item:idx for idx, item in enumerate(dataset.song_id.unique())}

        user_item_train = [(user, item) for user, item in zip(self.train.msno, self.train.song_id)]
        user_item_validation = [(user, item) for user, item in zip(self.validation.msno, self.validation.song_id)]
        user_item_test = [(user, item) for user, item in zip(self.test.msno, self.test.song_id)]

        ### make user-item sparse matrix for matrix factorization.
        rows = list()
        cols = list()
        data = list()

        for idx in self.train.index:
            user = self.train.loc[idx, 'msno']
            item = self.train.loc[idx, 'song_id']
            target = self.train.loc[idx, 'target']

            rows.append(user2idx[user])
            cols.append(item2idx[item])
            data.append(target) 
        
        train_matrix = csr_matrix((data, (rows, cols)), shape=(len(user2idx), len(item2idx)), dtype=float)

        return (user_item_train, user_item_validation, user_item_test), (user2idx, item2idx), train_matrix

    def train_nn(self, x_train, y_train, x_validation=None, y_validation=None, checkpint_dir='./checkpoints'):
        """Train neural network model.
        
        Keras style neural netowrk model training. 
        
        Args:
            x_train (ndarray): feature data in train dataset.
            y_train (ndarray): target data in train dataset.
            x_validation (ndarray): feature data in validation dataset.
            y_validation (ndarray): target data in validation dataset.
            checkpint_dir (str): Directory path to save model weights.
        """

        ### Model Summary
        temp_input = keras.Input(shape=(x_train.shape[1]))
        self.model_nn(temp_input)
        print(self.model_nn.summary())

        optimizer = keras.optimizers.Adam(lr=0.001)
        self.model_nn.compile(
            optimizer=optimizer, 
            loss=keras.losses.binary_crossentropy,
            metrics=[keras.metrics.AUC(name='auc')]
        )


        ### Load checkpoint or train model.
        filename = 'model-nn-weight.h5' if x_validation is None else 'partial-model-nn-weight.h5'
        filename = os.path.join(checkpint_dir, filename)
        if os.path.exists(filename):
            self.model_nn.load_weights(filename)
        else:
            if x_validation is not None:
                history = self.model_nn.fit(
                    x=x_train, 
                    y=y_train, 
                    batch_size=512, 
                    epochs=10, 
                    verbose=1, 
                    validation_data=(x_validation, y_validation),
                )
            else:
                history = self.model_nn.fit(
                    x=x_train, 
                    y=y_train, 
                    batch_size=512, 
                    epochs=10, 
                    verbose=1, 
                    validation_split=0.2
                )

            self.model_nn.save_weights(filename)
            self._draw_history(history, x_validation is None)
    
    def train_nmf(self, user_item_matrix, full=False, checkpint_dir='./checkpoints'):
        """Train matric factorization model.
        
        scikit-learn NMF model fitting. 
        
        Args:
            user_item_matrix (csr_matrix): user item sparse matrix.
        Return:
            W (csr_matrix): user-components matrix
            H (csr_matirx): component-item matrix
        """

        filename = 'model-nmf.joblib' if full else 'partial-model-nmf.joblib'
        filename = os.path.join(checkpint_dir, filename)
        if os.path.exists(filename):
            self.model_nmf = joblib.load(filename)
        else:
            self.model_nmf.fit(user_item_matrix)
            joblib.dump(self.model_nmf, filename)

        W = self.model_nmf.transform(user_item_matrix)
        H = self.model_nmf.components_

        return W, H

    def predict_nn(self, x):
        """Predict target using neural network model.
        
        Predict target values using neural network model. 
        
        Args:
            x (ndarray): feature data.
        Return:
            prediction (ndarray): prediction data.
        """

        prediction = self.model_nn.predict(x)
        return prediction

    def calculate_auc(self, real, prediction):
        """Calcualte AUC score.
        
        Calculate AUC score based on real and prediction data.
        
        Args:
            real (ndarray): real data.
            prediction (ndarray): prediction data.
        Return:
            score (ndarray): aud score.
        """

        score = roc_auc_score(real, prediction)
        return score

    def predict_nmf(self, x, W, H, user2idx, song2idx):
        """Predict target using matrix factorization model.
        
        Predict target values using matrix factorization model. 
        
        Args:
            x (ndarray): user item pair
            W (csr_matrix): user-components matrix
            H (csr_matirx): component-item matrix
            user2idx (dict): user to index dictionary
            item2idx (dict): item to index dictionary
        Return:
            prediction (ndarray): prediction data.
        """

        prediction = np.zeros(shape=(len(x), 1), dtype=float)

        users = np.array([user2idx[user] for user, _ in x])
        items = np.array([song2idx[song] for _, song in x])

        for idx in tqdm(range(len(x))):
            user = users[idx]
            item = items[idx]

            rating = max(0, min(np.dot(W[user, :], H[:, item]), 1))
            prediction[idx, 0] = rating
        
        return prediction

    def combine_predictions(self, predictions, weights):
        """Combine predictions drawn by models.
        
        Weighted sum of model predictions. 
        
        Args:
            predictions (list): list of predictions
            weights (list): weights for each prediction 
        Return:
            prediction (ndarray): prediction data.
        """

        prediction = np.zeros_like(predictions[0], dtype=float)

        for p, w in zip(predictions, weights):
            prediction += (p * w)
        
        return prediction

if __name__ == '__main__':
    recommender = RecommendationWorker(data_dirname='./data', font_path='./static/fonts/D2Coding.ttc')

    print('***** Load Dataset for Nerual Network Model *****')
    (x_train, y_train), (x_validation, y_validation), x_test = recommender.load_dataset_for_nn()
    
    print('Train Dataset Shape: {} {}'.format(x_train.shape, y_train.shape))
    print('Validation Dataset Shape: {} {}'.format(x_validation.shape, y_validation.shape))
    print('Test Dataset Shape: {}'.format(x_test.shape))

    print('\n***** Load Dataset for Matrix Factorization Model *****')
    (user_item_train, user_item_validation, user_item_test), (user2idx, item2idx), train_matrix = recommender.load_dataset_for_nmf()

    print('User Item Matrix Shape: {}'.format(train_matrix.shape))    

    print('\n***** Train Neural Network Model *****')
    recommender.train_nn(x_train, y_train, x_validation, y_validation)
    prediction_nn = recommender.predict_nn(x_validation)
    auc_nn = recommender.calculate_auc(y_validation, prediction_nn)
    print('Only Neural Net Model AUC Score: {:.4f}'.format(auc_nn))

    print('\n***** Train Matrix Factorization Model *****')
    W, H = recommender.train_nmf(train_matrix)
    prediction_nmf = recommender.predict_nmf(user_item_validation, W, H, user2idx, item2idx)
    auc_nmf =  recommender.calculate_auc(y_validation, prediction_nmf)
    print('Only NMF Model Score: {:.4f}'.format(auc_nmf))
    
    print('\n***** Combining Prediction Results from Models *****')
    prediction_combined = recommender.combine_predictions([prediction_nn, prediction_nmf], [0.6, 0.4])
    auc_combined = recommender.calculate_auc(y_validation, prediction_combined)
    print('NN + NMF Model Score: {:.4f}'.format(auc_combined))

    print('\n***** Compare Models: Draw ROC Curve *****')
    predictions = [prediction_nn, prediction_nmf, prediction_combined]
    scores = [auc_nn, auc_nmf, auc_combined]
    names = ['nn', 'nmf', 'nn+nmf']
    recommender.plot_roc_curves(y_validation, predictions, scores, names)

    print('\n***** Make Submission *****')
    x_train_validation = np.vstack((x_train, x_validation))
    y_train_validation = np.vstack((y_train, y_validation))
    recommender.train_nn(x_train_validation, y_train_validation)

    rows = list()
    cols = list()
    data = list()
    for (user, item), target in zip(user_item_train, y_train):
        rows.append(user2idx[user])
        cols.append(item2idx[item])
        data.append(target[0])

    for (user, item), target in zip(user_item_validation, y_validation):
        rows.append(user2idx[user])
        cols.append(item2idx[item])
        data.append(target[0])
    
    train_validation_matrix = csr_matrix((data, (rows, cols)), shape=(len(user2idx), len(item2idx)), dtype=float)
    W, H = recommender.train_nmf(train_validation_matrix, full=True)

    prediction_nn = recommender.predict_nn(x_test)
    prediction_nmf = recommender.predict_nmf(user_item_test, W, H, user2idx, item2idx)
    prediction = recommender.combine_predictions([prediction_nn, prediction_nmf], [0.6, 0.4])

    submission = pd.read_csv('./data/sample_submission.csv')
    submission['target'] = prediction.reshape(-1)
    submission.to_csv('./data/submissioin.csv', index=False)
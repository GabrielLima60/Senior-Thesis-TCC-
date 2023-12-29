# This code is ran as as subprocess of TCC_Main, don't run this script on it's own.

import pandas as pd
import numpy as np
import time
import sys
import tracemalloc

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE


# This is a dataset about iris flowers information and classification of its species
df_iris = pd.read_csv('datasets i could use/iris.csv')

# This is a dataset about pacients physical health and classification of weather they have a low (0) or high (1) risk of heart disease
df_heart = pd.read_csv('datasets i could use/heart.csv')

df_breast = pd.read_csv('datasets i could use/breast.csv')

# This is a dataset of fetures of credid card transactions and classification of weather it's fraudulent (1) or not (0).
df_creditcard = pd.read_csv('datasets i could use/creditcard.csv')
df_creditcard = df_creditcard.head(50000)

class Analysis:

    def __init__(self, dataframe, technique, model='SVM'):
        self.prepare_data(dataframe)
        self.perform_analysis(technique, model)

    def prepare_data(self, dataframe):
        # Drop 'id' column and shuffle the dataframe
        if 'id' in dataframe.columns:
            dataframe.drop('id', axis=1, inplace=True)
        dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

        # Define X matrix and y column
        self.X = dataframe.iloc[:, :-1]
        self.y = dataframe.iloc[:, -1]

    def perform_analysis(self, technique, model):
        kf = KFold(n_splits=5)
        f1_scores = []

        for train_index, test_index in kf.split(self.X):
            self.split_data(train_index, test_index)

            if technique == 'PCA':
                self.apply_pca()
            elif technique == 'IncPCA':
                self.apply_ipca()
            elif technique == 'ICA':
                self.apply_ica()
            elif technique == 'LDA':
                self.apply_lda()
            elif technique == 'SMOTE':
                self.apply_smote()
            
            self.apply_normalization()

            f1 = self.select_model_and_get_f1(model)
            f1_scores.append(f1)

        self.f1_total = np.mean(f1_scores)

    def split_data(self, train_index, test_index):
        self.X_train, self.X_test = self.X.iloc[train_index], self.X.iloc[test_index]
        self.y_train, self.y_test = self.y.iloc[train_index], self.y.iloc[test_index]

    def apply_pca(self):
        pca = PCA(n_components=0.95)
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)

    def apply_ipca(self):
        pca = PCA()
        pca.fit(self.X_train)
        n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
        ipca = IncrementalPCA(n_components=n_components)
        self.X_train = ipca.fit_transform(self.X_train)
        self.X_test = ipca.transform(self.X_test)
    
    def apply_ica(self):
        n_components = int(self.X_train.shape[1]/2)
        ica = FastICA(n_components=n_components, random_state=42)
        self.X_train = ica.fit_transform(self.X_train)
        self.X_test = ica.transform(self.X_test)

    def apply_lda(self):
        lda = LinearDiscriminantAnalysis()
        self.X_train = lda.fit_transform(self.X_train, self.y_train)

        self.X_test = lda.transform(self.X_test)

    def apply_smote(self):
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)  


    def apply_normalization(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def select_model_and_get_f1(self, model):
        model_dict = {'Naive Bayes': GaussianNB(),
                      'SVM': SVC(kernel='linear'),
                      'MLP': MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=300),
                      'Tree': DecisionTreeClassifier(),
                      'KNN': KNeighborsClassifier(),
                      'LogReg': LogisticRegression(),
                      'GBC': GradientBoostingClassifier()
                     }
        

        return self.get_f1_score(model_dict[model])

    def get_f1_score(self, classifier):
        classifier.fit(self.X_train, self.y_train)
        self.y_pred = classifier.predict(self.X_test)
        return f1_score(self.y_test, self.y_pred, average='weighted')
    


# We get the parameters passed by the parent script
dataset = sys.argv[1]
technique = sys.argv[2]
model = sys.argv[3]

dataset_dict = {'df_iris': df_iris,
                'df_heart': df_heart,
                'df_breast': df_breast,
                'df_creditcard': df_creditcard
}

# We use tracemalloc snapshots to get the memory used immediately before the execution and after it. 
tracemalloc.start()
start_time = time.time()
start_snapshot = tracemalloc.take_snapshot()


a = Analysis(dataset_dict[dataset], technique, model)


end_snapshot = tracemalloc.take_snapshot()
end_time = time.time()
tracemalloc.stop()

# The memory usage is shown in Kibibytes (KiB)
diff_snapshot = end_snapshot.compare_to(start_snapshot, 'lineno')
memory_usage = sum(stat.size for stat in diff_snapshot)/1024

processing_time = end_time - start_time

result = [dataset, technique, model, a.f1_total, processing_time, memory_usage]
print(result)
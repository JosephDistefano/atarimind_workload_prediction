
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

def classification_eye_gaze(config):
    feature_matrix = config['processed_features_path'] + 'breakout' + '/feature_matrices/' + 'eye_ccl_feature_matrix.csv'
    data = pd.read_csv(feature_matrix)
    X = data.iloc[:,:-4]
    Y = data["ProbDistraction"]

    # Y[Y<0] = 0
    # plt.hist(Y,bins=100)
    # plt.show()
    Y[Y>0.1] = 1
    Y[Y<0.1] = 0
    min_class_count = min(np.bincount(Y))
    undersampler = RandomUnderSampler(sampling_strategy={label: min_class_count for label in np.unique(Y)})
    X_balanced_under, y_balanced_under = undersampler.fit_resample(X, Y)

    print(y_balanced_under[y_balanced_under==1].count())
    print(y_balanced_under[y_balanced_under==0].count())


    X_train, X_test, y_train, y_test = train_test_split(X_balanced_under, y_balanced_under, test_size=0.33, random_state=42) 
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier on the training data
    rf_classifier.fit(X_train, y_train)

    # Predict the labels of the test set
    y_pred = rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    return
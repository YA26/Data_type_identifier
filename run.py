# -*- coding: utf-8 -*-
from data_type_identifier import DataTypeIdentifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
import pandas as pd

"""
############################################
############# MAIN OBJECT ##################
############################################
"""
data_type_identifier = DataTypeIdentifier(LabelEncoder)


"""
############################################
######## DATA PREPROCESSING ################
############################################
"""
# 1-Loading data
categorical_numerical_data = pd.read_csv(".\data\data.csv", sep=",", index_col=False) 

# 2-Separating our features from our target variable
features        = categorical_numerical_data.iloc[:,:-1]
target_variable = categorical_numerical_data.iloc[:,-1]

# 3-Transposing our feature data frame to implement a column type analysis
features_transposed = features.T

# 4-Keeping the initial data type of every single feature
features_transposed = data_type_identifier.keep_initial_data_types(features_transposed)

# 5-Building our training set
features_and_target = data_type_identifier.build_final_set(features_transposed, target_variable)
X_train             = features_and_target["new_features"]
y_train             = features_and_target["target_variable_encoded"]
mappings            = data_type_identifier.get_target_variable_class_mappings() # 0 for categorical and 1 for numerical when this model was built 

# 6-Shuffling our data
X_train , y_train = shuffle(X_train, y_train) 


"""
############################################
############## TRAINING ####################
############################################
"""
data_type_identifier_model=data_type_identifier.sigmoid_neuron(X=X_train,
                                                               y=y_train,
                                                               path="./model_and_checkpoint/data_type_identifier.h5", 
                                                               epoch=75, 
                                                               validation_split=0.1, 
                                                               batch_size=10)  

"""
############################################
##############SAVING VARIABLES##############
############################################
"""
"""
data_type_identifier.save_variables("./saved_variables/mappings.pickle", mappings)
data_type_identifier.save_variables("./saved_variables/X_train.pickle", mappings)
data_type_identifier.save_variables("./saved_variables/y_train.pickle", mappings) 
"""

"""
############################################
################ TESTING ###################
############################################
"""
# 1- Loading important variables
mappings = data_type_identifier.load_variables("./saved_variables/mappings.pickle")

# 2- Loading the model
data_type_identifier_model=load_model("./model_and_checkpoint/data_type_identifier.h5")

# 3- Predictions on test set
X_test = pd.read_csv("./data/X_test.csv", sep=",")
y_and_labels_test = pd.read_csv("./data/y_test.csv", sep=",")
y_test = y_and_labels_test["Y"]
new_test_set_predictions = data_type_identifier.predict(X_test, mappings, data_type_identifier_model)

#4- Classification report
report = classification_report(y_true=y_test, y_pred=new_test_set_predictions, output_dict=True)
report = pd.DataFrame(report).transpose()
report.to_csv("./data/report.csv")


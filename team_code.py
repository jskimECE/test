# #!/usr/bin/env python

# # Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# # change or remove non-required functions, and add your own functions.

# ################################################################################
# #
# # Optional libraries and functions. You can change or remove them.
# #
# ################################################################################

# from helper_code import *
# import numpy as np, os, sys
# import pandas as pd
# import mne
# from sklearn.impute import SimpleImputer                  
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# ################################################################################
# #
# # Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
# #
# ################################################################################

# # Train your model.
# def train_challenge_model(data_folder, model_folder, verbose):
#     # Find the Challenge data.
#     if verbose >= 1:
#         print('Extracting features and labels from the Challenge data...')
        
#     patient_ids, data, label, features = load_challenge_data(data_folder)
#     num_patients = len(patient_ids)

#     if num_patients==0:
#         raise FileNotFoundError('No data is provided.')
        
#     # Create a folder for the model if it does not already exist.
#     os.makedirs(model_folder, exist_ok=True)
    
    
#     # Train the models.
#     if verbose >= 1:
#         print('Training the Challenge models on the Challenge data...')
    
#     data = pd.get_dummies(data)
        
#     # Define parameters for random forest classifier and regressor.
#     n_estimators   = 123  # Number of trees in the forest.
#     max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
#     random_state   = 789  # Random state; set for reproducibility.

#     # Impute any missing features; use the mean value by default.
#     imputer = SimpleImputer().fit(data)

#     # Train the models.
#     data_imputed = imputer.transform(data)
#     prediction_model = RandomForestClassifier(
#         n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(data_imputed, label.ravel())


#     # Save the models.
#     save_challenge_model(model_folder, imputer, prediction_model)

#     if verbose >= 1:
#         print('Done!')
        
# # Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# # arguments of this function.
# def load_challenge_model(model_folder, verbose):
#     print('Loading the model...')
#     filename = os.path.join(model_folder, 'model.sav')
#     return joblib.load(filename)

# def run_challenge_model(model, data_folder, verbose):
#     imputer = model['imputer']
#     prediction_model = model['prediction_model']

#     # Load data.
#     patient_ids, data, label, features = load_challenge_data(data_folder)
    
#     data = pd.get_dummies(data)
    
#     # Impute missing data.
#     data_imputed = imputer.transform(data)

#     # Apply model to data.
#     prediction_binary = prediction_model.predict(data_imputed)[:]
#     prediction_probability = prediction_model.predict_proba(data_imputed)[:, 1]

#     return patient_ids, prediction_binary, prediction_probability


# ################################################################################
# #
# # Optional functions. You can change or remove these functions and/or add new functions.
# #
# ################################################################################

# # Save your trained model.
# def save_challenge_model(model_folder, imputer, prediction_model):
#     d = {'imputer': imputer, 'prediction_model': prediction_model}
#     filename = os.path.join(model_folder, 'model.sav')
#     joblib.dump(d, filename, protocol=0)


#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
# 
################################################################################

from helper_code import *
import numpy as np, os, sys
import pandas as pd
# import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find the Challenge data.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
        
    patient_ids, data, label, features = load_challenge_data(data_folder)
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise FileNotFoundError('No data is provided.')
        
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # # Generate dummies and store the column names for consistency
    # data = pd.get_dummies(data)
    # columns = data.columns

    # # Save the column names for later use during inference
    # with open(os.path.join(model_folder, 'columns.txt'), 'w') as f:
    #     f.write("\n".join(columns))
        
    # # Define parameters for random forest classifier and regressor.
    # n_estimators   = 123  # Number of trees in the forest.
    # max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    # random_state   = 789  # Random state; set for reproducibility.

    # # Impute any missing features; use the mean value by default.
    # imputer = SimpleImputer().fit(data)
    data = data.dropna()
    # Separate features and target
    X = data.copy()  # Replace with actual target column name
    y = label.copy()  # Replace with actual target column name
    #comorbidities_columns = [f'comorbidity_adm___{i}' for i in range(1, 10)]
    combined_columns = [ 'agecalc_adm', 'sex_adm'] +['bcsverbal_adm'] +  ['muac_mm_adm'] + ['spo2onoxy_adm'] #+'height_cm_adm', comorbidities_columns
    X_group=  X[combined_columns]
    with open(os.path.join(model_folder, 'columns.txt'), 'w') as f:
        f.write("\n".join(combined_columns))
    # X_train, X_test, y_train, y_test = train_test_split(X_group, y, test_size=0.3, random_state=42)

    # Separate numerical and categorical columns
    numerical_features = X_group.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X_group.select_dtypes(include=['object']).columns

    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        #('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('scaler', StandardScaler()),  # Standardize features

    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        #('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent category
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode categorical variables
    ])

    # Combine the numerical and categorical transformers into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create a pipeline with preprocessing and a classifier
    # prediction_model = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('classifier', HistGradientBoostingClassifier(random_state=42,class_weight='balanced',loss='log_loss'))
    # ])
    rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    # gb_pipeline = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('classifier', GradientBoostingClassifier(random_state=42))
    # ])

    hgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced', loss='log_loss'))
    ])

    # Create an ensemble model using VotingClassifier
    ensemble_model = VotingClassifier(estimators=[
        ('rf', rf_pipeline),
        # ('gb', gb_pipeline),
        ('hgb', hgb_pipeline)
    ], voting='soft')  # Use 'soft' for probabilities or 'hard' for majority vote

    # Train the ensemble model
    
    # Train the model
    ensemble_model.fit(X_group, y.ravel())

    # # Train the models.
    # data_imputed = imputer.transform(data)
    # prediction_model = RandomForestClassifier(
    #     n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(data_imputed, label.ravel())
    imputer = None
    # Save the models.
    save_challenge_model(model_folder, imputer, ensemble_model)

    if verbose >= 1:
        print('Done!')
        
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    if verbose >= 1:
        print('Loading the model...')

    # Load the saved column names
    with open(os.path.join(model_folder, 'columns.txt'), 'r') as f:
        columns = f.read().splitlines()

    model = joblib.load(os.path.join(model_folder, 'model.sav'))
    model['columns'] = columns
    return model

def run_challenge_model(model, data_folder, verbose):
    # imputer = model['imputer']
    prediction_model = model['prediction_model']
    columns = model['columns']

    # Load data.
    patient_ids, data, label, features = load_challenge_data(data_folder)
    
    # data = pd.get_dummies(data)

    # # Align test data with training columns, filling missing columns with 0
    # data = data.reindex(columns=columns, fill_value=0)
    
    # # Impute missing data.
    # data_imputed = imputer.transform(data)

    # Apply model to data.
    data = data.dropna()
    # Separate features and target
    X = data.copy()  # Replace with actual target column name
    y = label.copy()  # Replace with actual target column name
    #comorbidities_columns = [f'comorbidity_adm___{i}' for i in range(1, 10)]
    
    combined_columns = [ 'agecalc_adm', 'sex_adm'] +['bcsverbal_adm'] +  ['muac_mm_adm'] + ['spo2onoxy_adm'] #+'height_cm_adm', comorbidities_columns
    X_group=  X[combined_columns]

    # X_train, X_test, y_train, y_test = train_test_split(X_group, y, test_size=0.3, random_state=42)

    # Separate numerical and categorical columns
    numerical_features = X_group.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X_group.select_dtypes(include=['object']).columns

    # Preprocessing for numerical features
    # numerical_transformer = Pipeline(steps=[
    #     #('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    #     ('scaler', StandardScaler()),  # Standardize features

    # ])

    # # Preprocessing for categorical features
    # categorical_transformer = Pipeline(steps=[
    #     #('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent category
    #     ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode categorical variables
    # ])

    # # Combine the numerical and categorical transformers into a single ColumnTransformer
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', numerical_transformer, numerical_features),
    #         ('cat', categorical_transformer, categorical_features)
    #     ]
    # )
    data_transformed = X_group.copy()
    # prediction_binary = prediction_model.predict(data_transformed)[:]
    # prediction_probability = prediction_model.predict_proba(data_transformed)[:, 1]
    prediction_binary = prediction_model.predict(data_transformed)[:]
    prediction_probability = prediction_model.predict_proba(data_transformed)[:, 1]
    #print(patient_ids, prediction_binary, prediction_probability)
    
    return patient_ids, prediction_binary, prediction_probability

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, prediction_model):
    d = {'imputer': imputer, 'prediction_model': prediction_model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

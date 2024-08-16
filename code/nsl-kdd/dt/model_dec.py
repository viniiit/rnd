# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np
# import matplotlib.pyplot as plt

# column_names = ["duration", "protocol_type", "service", "flag", "src_bytes","dst_bytes", "land", "wrong_fragment", "urgent", "hot","num_failed_logins", "logged_in", "num_compromised", "root_shell","su_attempted", "num_root", "num_file_creations", "num_shells","num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login","count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate","srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate","dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate","dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate","class","difficulty"]
# # Load training and test datasets
# train_data = pd.read_csv('NSL-KDDTrain+.txt', names=column_names, header=None)
# test_data = pd.read_csv('NSL-KDDTest+.txt', names=column_names, header=None)

# # List of features to remove
# features_to_remove = []

# # Remove the features from the training data
# train_data.drop(features_to_remove, axis=1,inplace=True)

# # Remove the features from the test data
# test_data.drop(features_to_remove, axis=1,inplace=True)

# # Separate features and target for training data
# X_train = train_data.iloc[:, :-2].values 
# y_train = train_data.iloc[:, -2]

# # Separate features and target for test data
# X_test = test_data.iloc[:, :-2].values
# y_test = test_data.iloc[:, -2]

# encoder = LabelEncoder()
# X_train[:, 1] = encoder.fit_transform(X_train[:, 1])
# X_train[:, 2] = encoder.fit_transform(X_train[:, 2])
# X_train[:, 3] = encoder.fit_transform(X_train[:, 3])

# # Preprocess the test set similarly

# X_test[:, 1] = encoder.fit_transform(X_test[:, 1])
# X_test[:, 2] = encoder.fit_transform(X_test[:, 2])
# X_test[:, 3] = encoder.fit_transform(X_test[:, 3])

# # Encode the labels
# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)

# # Add an "unknown" class if not already present
# if 'unknown' not in label_encoder.classes_:
#     classes = list(label_encoder.classes_) + ['unknown']
#     label_encoder.classes_ = np.array(classes)

# def transform_with_unknown_handling(encoder, labels):
#     unknown_class_index = np.where(encoder.classes_ == 'unknown')[0][0]
#     transformed_labels = []
#     for label in labels:
#         try:
#             transformed_label = encoder.transform([label])[0]
#         except ValueError:
#             # Assign the "unknown" class index for unseen labels
#             transformed_label = unknown_class_index
#         transformed_labels.append(transformed_label)
#     return np.array(transformed_labels)

# y_test_encoded = transform_with_unknown_handling(label_encoder, y_test)


# # Initialize the Decision Tree Classifier
# clf = DecisionTreeClassifier(random_state=42)
# # clf = DecisionTreeClassifier(
# #     max_depth=10,                  # Maximum depth of the tree
# #     min_samples_split=2,         # Minimum number of samples required to split an internal node
# #     min_samples_leaf=2,           # Minimum number of samples required to be at a leaf node
# #     criterion='entropy'           # The function to measure the quality of a split
# # )

# # Train the classifier on the training data
# clf.fit(X_train, y_train_encoded)

# # Predict on the test data
# y_pred = clf.predict(X_test)

# # Evaluate the model's performance on the test data
# accuracy = accuracy_score(y_test_encoded, y_pred)
# print(f'Test Accuracy: {accuracy}')

# # Assume clf is a trained Decision Tree Classifier
# importances = clf.feature_importances_

# features = train_data.columns[:-2]
# low_importance_features = features[importances < 0.001]
# print(low_importance_features)

# # Visualize feature importances

# plt.barh(features, importances)
# plt.xlabel("Feature Importance")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.gca().invert_yaxis()
# plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,make_scorer,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from random import randint


max_depth=None
min_samples_leaf= 1
min_samples_split= 3

# # Assuming column_names defined as earlier
column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty"]

# # Load datasets
train_data = pd.read_csv('NSL-KDDTrain+.txt', names=column_names, header=None)
test_data = pd.read_csv('NSL-KDDTest+.txt', names=column_names, header=None)
# # test_data = test_data[~test_data['class'].isin(['saint', 'mscan'])]
# # print(test_data)
# # Prepare training data

train_data['class'] = train_data['class'].apply(lambda x: 'attack' if x != 'normal' else x)
test_data['class'] = test_data['class'].apply(lambda x: 'attack' if x != 'normal' else x)

# # List of DoS attacks from the image
# dos_attacks = ["apache2", "smurf", "neptune", "back", "teardrop", "pod", "land", "mailbomb", "processtable", "udpstorm"]
# r2l_attacks = ["warezclient", "guess_password", "warezmaster", "imap", "ftp_write", "named", "multihop", "phf", "spy", "sendmail", "snmpgetattack", "snmpguess", "worm", "xsnoop", "xlock"]
# u2r_attacks = ["buffer_overflow", "httptunnel", "rootkit", "loadmodule", "perl", "xterm", "ps", "sqlattack"]
# probe_attacks = ["satan", "saint", "ipsweep", "portsweep", "nmap", "mscan"]

# # Function to replace class names
# def replace_class_names(class_name):
#     if class_name in dos_attacks:
#         return 'dos'
#     elif class_name in r2l_attacks:
#         return 'r2l'
#     elif class_name in u2r_attacks:
#         return 'u2r'
#     elif class_name in probe_attacks:
#         return 'probe'
#     elif class_name=="normal":
#         return 'normal'
#     else:
#         return 'unknown'

# # Apply the function to the 'class' column in your train and test data
# train_data['class'] = train_data['class'].apply(replace_class_names)
# test_data['class'] = test_data['class'].apply(replace_class_names)


X = train_data.iloc[:, :-2].values
y = train_data.iloc[:, -2]

# Prepare test data
X_test = test_data.iloc[:, :-2].values
y_test = test_data.iloc[:, -2]

encoder = LabelEncoder()
# Encoding categorical features
for i in [1, 2, 3]:
    X[:, i] = encoder.fit_transform(X[:, i])

# Encoding categorical features in the test set
for i in [1, 2, 3]:
    X_test[:, i] = encoder.fit_transform(X_test[:, i])

# Encoding labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# y_encoded = encoder.fit_transform(y)

if 'unknown' not in label_encoder.classes_:
    classes = list(label_encoder.classes_) + ['unknown']
    label_encoder.classes_ = np.array(classes)

def transform_with_unknown_handling(encoder, labels):
    unknown_class_index = np.where(encoder.classes_ == 'unknown')[0][0]
    transformed_labels = []
    for label in labels:
        try:
            transformed_label = encoder.transform([label])[0]
        except ValueError:
            # Assign the "unknown" class index for unseen labels
            transformed_label = unknown_class_index
        transformed_labels.append(transformed_label)
    return np.array(transformed_labels)

# Encoding labels in the test set
y_test_encoded = transform_with_unknown_handling(label_encoder, y_test)

# KFold cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
accuracies = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    clf = DecisionTreeClassifier( max_depth=max_depth, min_samples_leaf= min_samples_leaf, min_samples_split=min_samples_split, random_state=42)


    clf.fit(X_train, y_train)
    y_pred_val = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    accuracies.append(accuracy)

# Calculate average accuracy from all folds
average_accuracy = np.mean(accuracies)
print(f'Average CV Accuracy: {average_accuracy}')

# Final model training on all training data
clf.fit(X, y_encoded)

# # Predict on the test data
y_pred_test = clf.predict(X_test)
binary_predictions= y_pred_test

cm = confusion_matrix(y_test_encoded, y_pred_test) #attack=negative, normal=positive
n_classes = cm.shape[0]

class_labels = label_encoder.classes_  # This should correspond to the labels used to encode y

# Dictionary to hold class labels with their FPR and FNR
class_fpr_fnr = {}

for i in range(n_classes):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (FP + FN + TP)

    fpr = FP / float(FP + TN) if (FP + TN) != 0 else 0
    fnr = FN / float(FN + TP) if (FN + TP) != 0 else 0
    
    # Assigning FPR and FNR to class labels
    class_fpr_fnr[class_labels[i]] = {'FPR': fpr, 'FNR': fnr}

# Print or return the dictionary
for label, rates in class_fpr_fnr.items():
    print(f"Class: {label}, FPR: {rates['FPR']:.4f}, FNR: {rates['FNR']:.4f}")


# # Evaluate the model's performance on the test data
test_accuracy = accuracy_score(y_test_encoded, y_pred_test)
print(max_depth)
print(min_samples_leaf)
print(min_samples_split)
print(f'Test Accuracy: {test_accuracy}')

print("F1 Score:",f1_score(y_test_encoded, y_pred_test, average='micro'))


####################################################################################################

max_depth=20
min_samples_leaf= 1
min_samples_split= 2


# # Assuming column_names defined as earlier
column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty"]

# # Load datasets
train_data = pd.read_csv('NSL-KDDTrain+.txt', names=column_names, header=None)
test_data = pd.read_csv('NSL-KDDTest+.txt', names=column_names, header=None)

# List of DoS attacks from the image
dos_attacks = ["apache2", "smurf", "neptune", "back", "teardrop", "pod", "land", "mailbomb", "processtable", "udpstorm"]
r2l_attacks = ["warezclient", "guess_password", "warezmaster", "imap", "ftp_write", "named", "multihop", "phf", "spy", "sendmail", "snmpgetattack", "snmpguess", "worm", "xsnoop", "xlock"]
u2r_attacks = ["buffer_overflow", "httptunnel", "rootkit", "loadmodule", "perl", "xterm", "ps", "sqlattack"]
probe_attacks = ["satan", "saint", "ipsweep", "portsweep", "nmap", "mscan"]

# Function to replace class names
def replace_class_names(class_name):
    if class_name in dos_attacks:
        return 'dos'
    elif class_name in r2l_attacks:
        return 'r2l'
    elif class_name in u2r_attacks:
        return 'u2r'
    elif class_name in probe_attacks:
        return 'probe'
    elif class_name=="normal":
        return 'normal'
    else:
        return 'unknown'

# Apply the function to the 'class' column in your train and test data
train_data['class'] = train_data['class'].apply(replace_class_names)
test_data['class'] = test_data['class'].apply(replace_class_names)

train_data['is_attack'] = (train_data['class'] == 'normal').astype(int)
test_data['is_attack']=binary_predictions

flip_percentage = 0.35
indices_to_flip = np.random.choice(train_data.index, size=int(len(train_data) * flip_percentage), replace=False)
train_data.loc[indices_to_flip, 'is_attack'] = 1 - train_data.loc[indices_to_flip, 'is_attack']

X = train_data.drop(['class', 'difficulty'], axis=1)
X_test = test_data.drop(['class', 'difficulty'], axis=1)

y = train_data.iloc[:, -3]
y_test = test_data.iloc[:, -3]

encoder = LabelEncoder()
# Encoding categorical features
for i in [1, 2, 3]:
    X.iloc[:, i] = encoder.fit_transform(X.iloc[:, i])

# Encoding categorical features in the test set
for i in [1, 2, 3]:
    X_test.iloc[:, i] = encoder.fit_transform(X_test.iloc[:, i])

# Encoding labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# y_encoded = encoder.fit_transform(y)

if 'unknown' not in label_encoder.classes_:
    classes = list(label_encoder.classes_) + ['unknown']
    label_encoder.classes_ = np.array(classes)

def transform_with_unknown_handling(encoder, labels):
    unknown_class_index = np.where(encoder.classes_ == 'unknown')[0][0]
    transformed_labels = []
    for label in labels:
        try:
            transformed_label = encoder.transform([label])[0]
        except ValueError:
            # Assign the "unknown" class index for unseen labels
            transformed_label = unknown_class_index
        transformed_labels.append(transformed_label)
    return np.array(transformed_labels)

# Encoding labels in the test set
y_test_encoded = transform_with_unknown_handling(label_encoder, y_test)

# KFold cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
accuracies = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    clf = DecisionTreeClassifier( max_depth=max_depth, min_samples_leaf= min_samples_leaf, min_samples_split=min_samples_split, random_state=42)


    clf.fit(X_train, y_train)
    y_pred_val = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    accuracies.append(accuracy)

# Calculate average accuracy from all folds
average_accuracy = np.mean(accuracies)
print(f'Average CV Accuracy: {average_accuracy}')

# Final model training on all training data
clf.fit(X, y_encoded)

# # Predict on the test data
y_pred_test = clf.predict(X_test)

cm = confusion_matrix(y_test_encoded, y_pred_test) #attack=negative, normal=positive
n_classes = cm.shape[0]

class_labels = label_encoder.classes_  # This should correspond to the labels used to encode y

# Dictionary to hold class labels with their FPR and FNR
class_fpr_fnr = {}

for i in range(n_classes):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (FP + FN + TP)

    fpr = FP / float(FP + TN) if (FP + TN) != 0 else 0
    fnr = FN / float(FN + TP) if (FN + TP) != 0 else 0
    
    # Assigning FPR and FNR to class labels
    class_fpr_fnr[class_labels[i]] = {'FPR': fpr, 'FNR': fnr}

# Print or return the dictionary
for label, rates in class_fpr_fnr.items():
    print(f"Class: {label}, FPR: {rates['FPR']:.4f}, FNR: {rates['FNR']:.4f}")


# # Evaluate the model's performance on the test data
test_accuracy = accuracy_score(y_test_encoded, y_pred_test)
print(max_depth)
print(min_samples_leaf)
print(min_samples_split)
print(f'Test Accuracy: {test_accuracy}')

print("F1 Score:",f1_score(y_test_encoded, y_pred_test, average='micro'))




# # Assume clf is a trained Decision Tree Classifier
# # importances = clf.feature_importances_

# # features = train_data.columns[:-2]
# # low_importance_features = features[importances < 0.001]
# # print(low_importance_features)

# # # Visualize feature importances

# # plt.barh(features, importances)
# # plt.xlabel("Feature Importance")
# # plt.ylabel("Feature")
# # plt.tight_layout()
# # plt.gca().invert_yaxis()
# # plt.show()

#Define the parameter grid to search


def custom_score(y_true, y_pred, **kwargs):
    # Compute F1-score, used for GridSearch optimization
    f1 = f1_score(y_true, y_pred, average='macro')
    # classes = kwargs.get('labels', [])
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred,labels=unique_labels)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Calculate FPR and FNR for each class
    FPR = [fp / (fp + tn) if (fp + tn) != 0 else 0 for fp, tn in zip(FP, TN)]
    FNR = [fn / (fn + tp) if (fn + tp) != 0 else 0 for fn, tp in zip(FN, TP)]
    ACC = accuracy_score(y_true, y_pred)

    # Retrieve class labels from the encoder (passed via kwargs)
    print(f"Accuracy: {ACC}")
    print("Class-specific metrics:")
    for i, cls in enumerate(unique_labels):
        print(f"Class {cls} - FPR: {FPR[i]:.4f}, FNR: {FNR[i]:.4f}")
    print(f"F1-score (Micro): {f1}")
    # Return F1-score because GridSearchCV optimizes based on this
    return f1



param_grid = {
    'max_depth': [None, 10, 20,25, 30, 40, 50],
    'min_samples_split': [2,3,4, 5, 10],
    'min_samples_leaf': [1, 2,3,4,5]
}

clf = DecisionTreeClassifier(random_state=42)
custom_scorer = make_scorer(custom_score, greater_is_better=True,labels=label_encoder.classes_)

f1_scorer = make_scorer(f1_score, average='micro')  

# Initialize the GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring=custom_scorer)

# Fit the grid search to the data
grid_search.fit(X, y_encoded)

# Print the best parameters and the corresponding score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Use the best estimator to make predictions
y_pred = grid_search.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test_encoded, y_pred))
print("Test F1 Score:", f1_score(y_test_encoded, y_pred, average='micro'))


# # Define the parameter distribution to sample from
# param_dist = {
#     'max_depth': [None] + list(range(10, 50, 5)),
#     'min_samples_split': randint(2, 11),
#     'min_samples_leaf': randint(1, 5)
# }

# # Initialize the Decision Tree Classifier
# clf = DecisionTreeClassifier(random_state=42)

# # Initialize the RandomizedSearchCV
# random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist,
#                                    n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1, scoring='accuracy')

# # Fit the random search to the data
# random_search.fit(X, y_encoded)

# # Print the best parameters and the corresponding score
# print("Best Parameters:", random_search.best_params_)
# print("Best Score:", random_search.best_score_)

# # Use the best estimator to make predictions
# y_pred = random_search.predict(X_test)
# print("Test Accuracy:", accuracy_score(y_test_encoded, y_pred))

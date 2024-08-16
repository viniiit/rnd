import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE


b_learning_rate=0.001
b_epochs=100
b_batch_size=64
b_patience=7
b_h_l=0

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        

experiment_start_time = datetime.datetime.now()

# Assuming column_names defined as earlier
column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty"]

# Assuming you have the following files
# train_file_path = 'NSL-KDDTrain+.txt'
# test_file_path = 'NSL-KDDTest+.txt'
# train_file_path = 'train.txt'
# test_file_path = 'test.txt'

# Load the datasets
train_data = pd.read_csv('NSL-KDDTrain+.txt', names=column_names, header=None)
test_data = pd.read_csv('NSL-KDDTest+.txt', names=column_names, header=None)

train_data['class'] = train_data['class'].apply(lambda x: 'attack' if x != 'normal' else x)
test_data['class'] = test_data['class'].apply(lambda x: 'attack' if x != 'normal' else x)

# Assuming the last column is difficulty and the second last is attack type
X = train_data.iloc[:, :-2].values
encoder = LabelEncoder()
X[:, 1] = encoder.fit_transform(X[:, 1])
X[:, 2] = encoder.fit_transform(X[:, 2])
X[:, 3] = encoder.fit_transform(X[:, 3])
y = train_data.iloc[:, -2]


# Preprocess the test set similarly
X_test = test_data.iloc[:, :-2].values
X_test[:, 1] = encoder.fit_transform(X_test[:, 1])
X_test[:, 2] = encoder.fit_transform(X_test[:, 2])
X_test[:, 3] = encoder.fit_transform(X_test[:, 3])
y_test = test_data.iloc[:, -2]


# Encode the labels
attack_encoder = LabelEncoder()
y_encoded = attack_encoder.fit_transform(y)

# Add an "unknown" class if not already present
if 'unknown' not in attack_encoder.classes_:
    classes = list(attack_encoder.classes_) + ['unknown']
    attack_encoder.classes_ = np.array(classes)

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

y_test_encoded = transform_with_unknown_handling(attack_encoder, y_test)



# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_test_scaled = scaler.transform(X_test)

normalizer = Normalizer(norm='l2')

# Fit and transform the training data
X = normalizer.fit_transform(X)
X_test = normalizer.fit_transform(X_test)

# Split the training data to create a validation set
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded,  test_size=0.2, random_state=42)


def build_bin_model(input_dim,num_classes):
    global b_h_l
    model = Sequential()

    model.add(Dense(1024, input_dim=input_dim, activation='relu'))
    b_h_l+=1  # Input layer to 1st fully connected layer
    # model.add(Dense(1024, activation=None))  # 1st to 2nd layer fully connected with 1024 units
    model.add(BatchNormalization()) 
    b_h_l+=1 # Batch normalization after 1st layer
    # model.add(Dense(1024, activation='relu'))  # 2nd to 3rd layer dropout with 1024 units
    model.add(Dropout(0.01))
    b_h_l+=1  # Dropout after 2nd layer
    model.add(Dense(768, activation=None)) 
    b_h_l+=1 # 3rd to 4th layer fully connected with 768 units
    # model.add(Dense(768, activation=None))  # 4th to 5th layer batch normalization with 768 units
    model.add(BatchNormalization())
    b_h_l+=1  # Batch normalization after 4th layer
    model.add(Dense(768, activation=None))  
    b_h_l+=1 
    model.add(Dense(1, activation='sigmoid'))  # 5th to output layer of 1 unit

    optimizer= Adam(learning_rate=b_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

num_classes = len(np.unique(y_encoded))



# You also need to pass the number of features (input_dim) your model should expect
input_dim = X_train.shape[1]

# Now build the model
model = build_bin_model(input_dim,num_classes)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)

class_weight_dict[0]=1.5
class_weight_dict[1]=0.8

# # Display the model's architecture
# model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=b_patience, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val), class_weight=class_weight_dict,
                    epochs=b_epochs, batch_size=b_batch_size, verbose=1,callbacks=[early_stopping])

# print(history.history.keys())

bin_model_summary_str = []
model.summary(print_fn=lambda x: bin_model_summary_str.append(x))

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

b_train_accuracy=np.mean(train_accuracy)
b_val_accuracy=np.mean(val_accuracy)

binary_predictions =(model.predict(X_test) > 0.5).astype(int)

cm = confusion_matrix(y_test_encoded, binary_predictions) #attack=negative, normal=positive
# print("Confusion Matrix:\n", cm)
fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=attack_encoder.classes_, yticklabels=attack_encoder.classes_)
ax1.set_xlabel('Predicted labels')
ax1.set_ylabel('True labels')
ax1.set_title('Binary Confusion Matrix')
ax1.xaxis.set_ticklabels(attack_encoder.classes_)
ax1.yaxis.set_ticklabels(attack_encoder.classes_)


file_name1 = f"expr-{b_learning_rate}-{b_epochs}-{b_batch_size}-{b_patience}-{b_h_l}"

fig1.savefig(f"b-cm-{file_name1}.png",dpi=300)

n_classes = cm.shape[0]

class_labels = attack_encoder.classes_  # This should correspond to the labels used to encode y

# Dictionary to hold class labels with their FPR and FNR
bin_class_fpr_fnr = {}

for i in range(n_classes):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (FP + FN + TP)

    fpr = FP / float(FP + TN) if (FP + TN) != 0 else 0
    fnr = FN / float(FN + TP) if (FN + TP) != 0 else 0
    
    # Assigning FPR and FNR to class labels
    bin_class_fpr_fnr[class_labels[i]] = {'FPR': fpr, 'FNR': fnr}

adjusted_target_names = [name for i, name in enumerate(attack_encoder.classes_) if i in np.unique(y_test_encoded)]
bin_report = classification_report(y_test_encoded,binary_predictions, target_names=adjusted_target_names, labels=np.unique(y_test_encoded), zero_division=0, output_dict=False)

test_results = model.evaluate(X_test, y_test_encoded, verbose=1)
# print(test_results)
# print(f"Test Loss: {test_results[0]}")
bin_test_accuracy = test_results[1]


# Plot training and validation accuracy values
fig, ax2 = plt.subplots()
ax2.plot(history.history['accuracy'], label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')

# Set labels and title
ax2.set_title('Binary Training and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()



fig.savefig(f"b-plot-{file_name1}.png")




learning_rate=0.001
epochs=500
batch_size=64
patience=5
h_l=0
# learning_rate=0.1
# epochs=500
# batch_size=64

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
        


# Assuming column_names defined as earlier
column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty"]

# Assuming you have the following files
# train_file_path = 'NSL-KDDTrain+.txt'
# test_file_path = 'NSL-KDDTest+.txt'
# train_file_path = 'train.txt'
# test_file_path = 'test.txt'

# Load the datasets
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

# # Assuming 'data' is your DataFrame and 'class' is the target column
# train_data['is_attack'] = (train_data['class'] != 'normal').astype(int)
# test_data['is_attack'] = (test_data['class'] != 'normal').astype(int)
test_data['is_attack']=binary_predictions

# # Percentage of values to flip in the 'is_attack' column
# flip_percentage = 0.15
# indices_to_flip = np.random.choice(train_data.index, size=int(len(train_data) * flip_percentage), replace=False)

# # Flip the values in 'is_attack'
# train_data.loc[indices_to_flip, 'is_attack'] = 1 - train_data.loc[indices_to_flip, 'is_attack']



# Now split your data into features and labels, and ensure 'is_attack' is included in the feature set
# x = train_data.drop(['class', 'difficulty'], axis=1).values
# y = data['class'].values

# train_data['class'] = train_data['class'].apply(lambda x: 'attack' if x != 'normal' else x)
# test_data['class'] = test_data['class'].apply(lambda x: 'attack' if x != 'normal' else x)

X_train=train_data.drop(['class', 'difficulty'], axis=1)
y=train_data['class'].values

encoder = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    X_train[col] = encoder.fit_transform(X_train[col])
    test_data[col] = encoder.transform(test_data[col])

X_train=X_train.values

smote = SMOTE(random_state=42)
X_train, y= smote.fit_resample(X_train, y)

X_train=pd.DataFrame(X_train, columns=column_names[:-2])
start_index = 0  # replace with actual start index
end_index = start_index + len(binary_predictions) - 1

# Assign predictions to the specific segment
X_train.loc[start_index:end_index, 'is_attack'] = binary_predictions

y=pd.DataFrame(y, columns=[column_names[-2]])


original_length = len(binary_predictions)
# Manually set `is_attack` for new SMOTE-generated records
for index in range(original_length, len(X_train)):
    X_train.at[index, 'is_attack'] = 1 if y.at[index, 'class'] == 'normal' else 0

y=y.values

def flip_labels(data, flip_percentage):

    # Percentage of values to flip in the 'is_attack' column
    indices_to_flip = np.random.choice(data.index, size=int(len(data) * flip_percentage), replace=False)

    # Flip the values in 'is_attack'
    data.loc[indices_to_flip, 'is_attack'] = 1 - data.loc[indices_to_flip, 'is_attack']

    # for label in [0, 1]:
    #     # Find indices where is_attack equals the current label
    #     indices = data[data['is_attack'] == label].index
    #     # Calculate how many indices to flip
    #     n_flip = int(len(indices) * flip_percentage/2)
    #     # Select random indices to flip
    #     flip_indices = np.random.choice(indices, size=n_flip, replace=False)
    #     # Flip the selected indices
    #     data.loc[flip_indices, 'is_attack'] = 1 - label

flip_percentage = 0.25
# Flip a percentage of the SMOTE-generated `is_attack` values
flip_labels(X_train.loc[original_length:], flip_percentage)


# Features excluding 'is_attack' and 'class'
X_main = X_train.drop(['is_attack'], axis=1).values
X_main_test = test_data.drop(['class', 'difficulty', 'is_attack'], axis=1).values

# 'is_attack' feature alone
X_is_attack = X_train[['is_attack']].values
X_is_attack_test = test_data[['is_attack']].values

# Encode main categorical features
# encoder = LabelEncoder()
# for col in ['protocol_type', 'service', 'flag']:
#     X_main[col] = encoder.fit_transform(X_main[col])
#     X_main_test[col] = encoder.transform(X_main_test[col])

# Normalize the features
normalizer = Normalizer(norm='l2')
X_main = normalizer.fit_transform(X_main)
X_main_test = normalizer.transform(X_main_test)
X_is_attack = normalizer.fit_transform(X_is_attack)
X_is_attack_test = normalizer.transform(X_is_attack_test)




# # Assuming the last column is difficulty and the second last is attack type
# # X = train_data.iloc[:, :-2].values
# X = train_data.drop(['class', 'difficulty'], axis=1).values
# encoder = LabelEncoder()
# X[:, 1] = encoder.fit_transform(X[:, 1])
# X[:, 2] = encoder.fit_transform(X[:, 2])
# X[:, 3] = encoder.fit_transform(X[:, 3])
# y = train_data.iloc[:, -3]


# # Preprocess the test set similarly
# # X_test = test_data.iloc[:, :-2].values
# X_test = test_data.drop(['class', 'difficulty'], axis=1).values
# X_test[:, 1] = encoder.fit_transform(X_test[:, 1])
# X_test[:, 2] = encoder.fit_transform(X_test[:, 2])
# X_test[:, 3] = encoder.fit_transform(X_test[:, 3])
y_test = test_data.iloc[:, -3]


# Encode the labels
attack_encoder = LabelEncoder()
y_encoded = attack_encoder.fit_transform(y)

# Add an "unknown" class if not already present
if 'unknown' not in attack_encoder.classes_:
    classes = list(attack_encoder.classes_) + ['unknown']
    attack_encoder.classes_ = np.array(classes)

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

y_test_encoded = transform_with_unknown_handling(attack_encoder, y_test)


# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_test_scaled = scaler.transform(X_test)

normalizer = Normalizer(norm='l2')

# # Fit and transform the training data
# X = normalizer.fit_transform(X)
# X_test = normalizer.fit_transform(X_test)



# # Split the training data to create a validation set
# X_train, X_val, y_train, y_val = train_test_split(X, y_encoded,  test_size=0.2, random_state=42)

X_main_train, X_main_val, y_train, y_val = train_test_split(
    X_main, y_encoded, test_size=0.2, random_state=42)

# Split is_attack feature similarly using the same random_state
X_is_attack_train, X_is_attack_val = train_test_split(
    X_is_attack, test_size=0.2, random_state=42)


# # Combine your main features and 'is_attack' feature
# X_combined = np.hstack((X_main_train, X_is_attack_train))

# smote = SMOTE(random_state=42)
# X_train_resampled, y_train = smote.fit_resample(X_combined, y_train)

# # After resampling, split the features back
# X_main_train = X_train_resampled[:, :-1]  # Assuming 'is_attack' is the last column
# X_is_attack_train = X_train_resampled[:, -1]




# sampling_strategy = {class_label: 10000 for class_label, count in Counter(y_train).items() if count < 400}  # Ensuring at least 50 samples for the smallest classes
# over = RandomOverSampler(sampling_strategy=sampling_strategy)
# under = RandomUnderSampler(sampling_strategy={11: 5000, 9: 5000})  # Reduce counts of largest classes

# pipeline = Pipeline([
#     ('o', over),
#     ('u', under)
# ])

# X_train, y_train = pipeline.fit_resample(X_train, y_train)
# X_train_resampled, y_train_resampled = over.fit_resample([X_main_train,X_is_attack_train], y_train)

def build_mul_model(num_main_features, num_classes):
    global h_l
    
    # Main input
    main_input = Input(shape=(num_main_features,), name='main_input')
    x = Dense(1024, activation='relu')(main_input)
    h_l+=1
    x = BatchNormalization()(x)
    h_l+=1
    x = Dropout(0.01)(x)
    h_l+=1
    x = Dense(768, activation='relu')(x)
    h_l+=1
    
    # is_attack input
    is_attack_input = Input(shape=(1,), name='is_attack_input')
    y = Dense(32, activation='relu')(is_attack_input)
    y = Dense(16, activation='relu')(y)

    # Combine the outputs
    combined = concatenate([x, y])

    # Output layer
    output = Dense(num_classes, activation='sigmoid')(combined)  # Adjust activation based on task
    
    model = Model(inputs=[main_input, is_attack_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


num_classes = len(np.unique(y_encoded))



# # You also need to pass the number of features (input_dim) your model should expect
# input_dim = X_train.shape[1]

# # Now build the model
# model = build_model(input_dim,num_classes)


num_main_features = X_main.shape[1]
model = build_mul_model(num_main_features, num_classes)

# Display the model's architecture
model.summary()

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("weights:",class_weight_dict)


class_weight_dict[1] *= 1.4
class_weight_dict[2] *= 2.5
class_weight_dict[3] *= 20


early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

history= model.fit(
    [X_main_train, X_is_attack_train], y_train,
    validation_data=([X_main_val, X_is_attack_val], y_val), class_weight=class_weight_dict,
    epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping]
)

# history = model.fit(X_train, y_train,
#                     validation_data=(X_val, y_val), class_weight=class_weight_dict,
#                     epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping]
#                     )

# print(history.history.keys())

mul_model_summary_str = []
model.summary(print_fn=lambda x: mul_model_summary_str.append(x))

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

train_accuracy=np.mean(train_accuracy)
val_accuracy=np.mean(val_accuracy)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")


# # Make predictions with your model
# predictions = np.argmax(model.predict(X_test), axis=-1)
pred=model.predict([X_main_test, X_is_attack_test])
print(pred)
predictions=np.argmax(pred,axis=-1)

cm = confusion_matrix(y_test_encoded, predictions) #attack=negative, normal=positive
print("Confusion Matrix:\n", cm)
fig1, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=attack_encoder.classes_, yticklabels=attack_encoder.classes_)
ax3.set_xlabel('Predicted labels')
ax3.set_ylabel('True labels')
ax3.set_title('Confusion Matrix')
ax3.xaxis.set_ticklabels(attack_encoder.classes_)
ax3.yaxis.set_ticklabels(attack_encoder.classes_)


file_name = f"{flip_percentage}-{b_learning_rate}_{learning_rate}-{b_epochs}_{epochs}-{b_batch_size}_{batch_size}-{b_patience}_{patience}-{b_h_l}_{h_l}"

fig1.savefig(f"cm-{file_name}.png",dpi=300)
# plt.show()



n_classes = cm.shape[0]

class_labels = attack_encoder.classes_  # This should correspond to the labels used to encode y

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


# print(np.unique(y_test_encoded))
# print(attack_encoder.classes_)
# print(len(attack_encoder.classes_))  # This should give you the number of unique classes
# print(range(len(attack_encoder.classes_)))

adjusted_target_names = [name for i, name in enumerate(attack_encoder.classes_) if i in np.unique(y_test_encoded)]

report = classification_report(y_test_encoded, predictions, target_names=adjusted_target_names, labels=np.unique(y_test_encoded), zero_division=0, output_dict=False)

# Get the class names for the predictions and the actual values (if your version of Keras doesn't support predict_classes(), you'll need to use predict() and then convert the probabilities to class labels)
predicted_class_labels = attack_encoder.inverse_transform(predictions)
actual_class_labels = attack_encoder.inverse_transform(y_test_encoded)

# Identify the indices of the DoS attack instances in the actual test set
# dos_attack_indices = [index for index, label in enumerate(actual_class_labels) if label in dos_attacks]
# r2l_attacks_indices = [index for index, label in enumerate(actual_class_labels) if label in r2l_attacks]
# u2r_attacks_indices = [index for index, label in enumerate(actual_class_labels) if label in u2r_attacks]
# probe_attacks_indices = [index for index, label in enumerate(actual_class_labels) if label in probe_attacks]
# normal_indices = [index for index, label in enumerate(actual_class_labels) if label == 'normal']

dos_attack_indices = [index for index, label in enumerate(actual_class_labels) if label=='dos']
r2l_attacks_indices = [index for index, label in enumerate(actual_class_labels) if label=='r2l']
u2r_attacks_indices = [index for index, label in enumerate(actual_class_labels) if label=='u2r']
probe_attacks_indices = [index for index, label in enumerate(actual_class_labels) if label=='probe']
normal_indices = [index for index, label in enumerate(actual_class_labels) if label == 'normal']

# Count the number of correct predictions for DoS attack instances
# correct_dos_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in dos_attack_indices)
# correct_r2l_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in r2l_attacks_indices)
# correct_u2r_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in u2r_attacks_indices)
# correct_probe_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in probe_attacks_indices)
# correct_normal_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in normal_indices)

correct_dos_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in dos_attack_indices)
correct_r2l_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in r2l_attacks_indices)
correct_u2r_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in u2r_attacks_indices)
correct_probe_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in probe_attacks_indices)
correct_normal_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in normal_indices)


# Calculate the total number of DoS attack instances
total_dos_attacks = len(dos_attack_indices)
total_r2l_attacks = len(r2l_attacks_indices)
total_u2r_attacks = len(u2r_attacks_indices)
total_probe_attacks = len(probe_attacks_indices)
total_normal = len(normal_indices)

print(total_dos_attacks,total_r2l_attacks, total_u2r_attacks, total_probe_attacks)
print(correct_dos_predictions,correct_r2l_predictions,correct_u2r_predictions,correct_probe_predictions)
# Calculate the accuracy for DoS attack predictions
dos_attack_accuracy = correct_dos_predictions / total_dos_attacks if total_dos_attacks > 0 else 0
r2l_attack_accuracy = correct_r2l_predictions / total_r2l_attacks if total_r2l_attacks > 0 else 0
u2r_attack_accuracy = correct_u2r_predictions / total_u2r_attacks if total_u2r_attacks > 0 else 0
probe_attack_accuracy = correct_probe_predictions / total_probe_attacks if total_probe_attacks > 0 else 0
normal_accuracy = correct_normal_predictions / total_normal if total_normal > 0 else 0

print(f"Accuracy for DoS attacks: {dos_attack_accuracy}")
print(f"Accuracy for R2L attacks: {r2l_attack_accuracy}")
print(f"Accuracy for U2R attacks: {u2r_attack_accuracy}")
print(f"Accuracy for Probe attacks: {probe_attack_accuracy}")
print(f"Accuracy for Normal: {normal_accuracy}")

# # Evaluate the model on the test data
# test_results = model.evaluate(X_test, y_test_encoded, verbose=1)

# Evaluate the model on the test data
test_results = model.evaluate([X_main_test, X_is_attack_test], y_test_encoded, verbose=1)


# print(test_results)
# print(f"Test Loss: {test_results[0]}")
print(f"Test Accuracy: {test_results[1]}")
print(report)

experiment_end_time = datetime.datetime.now()

# Analyze learning progress

# # Find epochs required for learning patterns and detecting attacks
# epochs_to_learn = None
# overfitting_epochs = None

# # Track training accuracy changes over epochs
# for epoch in range(1, epochs):  # Assuming epochs is defined earlier in your code
#     if epoch > 0 and train_accuracy[epoch] > train_accuracy[epoch - 1]:
#         epochs_to_learn = epoch + 1
#         break

# # Identify overfitting
# for epoch in range(1, epochs):
#     if epoch > 0 and val_accuracy[epoch] < val_accuracy[epoch - 1]:
#         overfitting_epochs = epoch
#         break


# # Display results
# print(f"Epochs required to learn patterns and detect attacks: {epochs_to_learn}")
# print(f"Epochs until overfitting occurs: {overfitting_epochs}")


# Plot training and validation accuracy values

fig, ax4 = plt.subplots()
ax4.plot(history.history['accuracy'], label='Training Accuracy')
ax4.plot(history.history['val_accuracy'], label='Validation Accuracy')

# Set labels and title
ax4.set_title('Training and Validation Accuracy')
ax4.set_xlabel('Epochs')
ax4.set_ylabel('Accuracy')
ax4.legend()

experiment_duration = experiment_end_time - experiment_start_time

start_time_formatted = experiment_start_time.strftime("%Y-%m-%d-%H-%M-%S")

# Show plot
# plt.show()

fig.savefig(f"plot-{file_name}.png")



with open(f"{file_name}.txt", "w") as file:
    file.write(f"Start Time: {start_time_formatted}\n")
    file.write(f"Total Runtime: {experiment_duration}\n\n")
    file.write(f"Experiment Details: binary\n")
    file.write(f"Learning Rate: {b_learning_rate}\n")
    file.write(f"Epochs: {b_epochs}\n")
    file.write(f"Patience: {b_patience}\n")
    file.write(f"Batch Size: {b_batch_size}\n")
    file.write(f"hidden_layers(binary): {b_h_l}\n")
    file.write(f"Train_accuracy: {b_train_accuracy}\n")
    file.write(f"Val_accuracy: {b_val_accuracy}\n")
    file.write(f"Accuracy: {bin_test_accuracy}\n")
    file.write("\nFPR and FNR by Class:\n")
    for label, rates in bin_class_fpr_fnr.items():
        file.write(f"Class: {label}, FPR: {rates['FPR']:.4f}, FNR: {rates['FNR']:.4f}\n")
    file.write("\nBinary Classification Report:\n")
    file.write(bin_report)
    file.write("\nbin Model Summary:\n")
    file.write("\n".join(bin_model_summary_str))


    file.write(f"\n\n  Experiment Details: multi\n")
    file.write(f"drup\n")
    file.write(f"Learning Rate: {learning_rate}\n")
    file.write(f"Epochs: {epochs}\n")
    file.write(f"Patience: {patience}\n")
    file.write(f"Batch Size: {batch_size}\n")
    file.write(f"hidden_layers(multi): {h_l}\n")
    file.write(f"flip percentage: {flip_percentage}\n")
    file.write(f"Val_accuracy: {val_accuracy}\n")
    file.write(f"Accuracy: {test_results[1]}\n")
    file.write(f"DOS Accuracy: {dos_attack_accuracy}\n")
    file.write(f"R2L Accuracy: {r2l_attack_accuracy}\n")
    file.write(f"U2R Accuracy: {u2r_attack_accuracy}\n")
    file.write(f"PROBE Accuracy: {probe_attack_accuracy}\n")
    file.write(f"Normal Accuracy: {normal_accuracy}\n")
    
    # file.write("Confusion Matrix:\n")
    # file.write(str(cm))
    file.write("\nFPR and FNR by Class:\n")
    for label, rates in class_fpr_fnr.items():
        file.write(f"Class: {label}, FPR: {rates['FPR']:.4f}, FNR: {rates['FNR']:.4f}\n")
    file.write("\nClassification Report:\n")
    file.write(report)
    file.write("\nMulti Model Summary:\n")
    file.write("\n".join(mul_model_summary_str))
   
print(f"Experiment details saved to {file_name}")

# File paths
text_file_path = f"{file_name}.txt"
confusion_matrix_image_path = f"cm-{file_name}.png"
graph_image_path = f"plot-{file_name}.png"
bin_confusion_matrix_image_path = f"b-cm-{file_name1}.png"
bin_graph_image_path = f"b-plot-{file_name1}.png"
pdf_path = f"pdf-{file_name}.pdf"

# Setup the PDF document
c = canvas.Canvas(pdf_path, pagesize=letter)
width, height = letter  # Get width and height of the page
styles = getSampleStyleSheet()

# Add text from a text file
y_position = height - 60  # Start 60 pixels down from the top

try:
    with open(text_file_path, 'r') as file:
        for line in file:
            p = Paragraph(line.strip(), style=styles["Normal"])
            w, h = p.wrap(width - 80, height)  # wrap the paragraph to width of the page
            if y_position < h + 50:  # Check if the paragraph fits before the image
                c.showPage()  # Start a new page if not enough space
                y_position = height - 60
            p.drawOn(c, 40, y_position - h)  # draw the paragraph on the canvas
            y_position -= (h + 10)  # move the cursor y_position for the next paragraph
except FileNotFoundError:
    print("Text file not found.")

# Add the binary confusion matrix image
if y_position < 350:  # Ensure there's enough space for the binary confusion matrix
    c.showPage()
    y_position = height - 60

try:
    binary_confusion_matrix = ImageReader(bin_confusion_matrix_image_path)
    c.drawImage(binary_confusion_matrix, 40, y_position - 350, width=520, height=300, preserveAspectRatio=True)
    y_position -= 350  # Adjust y_position after the binary confusion matrix
except Exception as e:
    print("Failed to load binary confusion matrix image:", e)

# Add the binary graph image
if y_position < 350:
    c.showPage()
    y_position = height - 60

try:
    binary_graph_image = ImageReader(bin_graph_image_path)
    c.drawImage(binary_graph_image, 40, y_position - 350, width=520, height=300, preserveAspectRatio=True)
    y_position -= 350  # Adjust y_position after the binary graph
except Exception as e:
    print("Failed to load binary graph image:", e)

# Add the main confusion matrix image
if y_position < 350:  # Ensure there's enough space for the main confusion matrix
    c.showPage()
    y_position = height - 60

try:
    main_confusion_matrix = ImageReader(confusion_matrix_image_path)
    c.drawImage(main_confusion_matrix, 40, y_position - 350, width=520, height=300, preserveAspectRatio=True)
    y_position -= 350  # Adjust y_position after the main confusion matrix
except Exception as e:
    print("Failed to load main confusion matrix image:", e)

# Add the main graph image
if y_position < 350:
    c.showPage()
    y_position = height - 60

try:
    main_graph_image = ImageReader(graph_image_path)
    c.drawImage(main_graph_image, 40, y_position - 350, width=520, height=300, preserveAspectRatio=True)
    y_position -= 350  # Adjust y_position after the main graph
except Exception as e:
    print("Failed to load main graph image:", e)

c.save()
print("PDF created at:", pdf_path)
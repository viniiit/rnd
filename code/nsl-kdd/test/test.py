import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
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

learning_rate=0.001
epochs=500
batch_size=128
patience=10

# learning_rate=0.1
# epochs=500
# batch_size=64

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


# List of DoS attacks from the image
dos_attacks = ["apache2", "smurf", "neptune", "back", "teardrop", "pod", "land", "mailbomb", "processtable", "udpstorm"]
r2l_attacks = ["warezclient", "guess_password", "warezmaster", "imap", "ftp_write", "named", "multihop", "phf", "spy", "sendmail", "snmpgetattack", "snmpguess", "worm", "xsnoop", "xlock"]
u2r_attacks = ["buffer_overflow", "httptunnel", "rootkit", "loadmodule", "perl", "xterm", "ps", "sqlattack"]
probe_attacks = ["satan", "saint", "ipsweep", "portsweep", "nmap", "mscan"]

# # Function to replace class names
# def replace_class_names(class_name):
#     if class_name in dos_attacks:
#         return 'DoS'
#     else:
#         return 'normal'

# # Apply the function to the 'class' column in your train and test data
# train_data['class'] = train_data['class'].apply(replace_class_names)
# test_data['class'] = test_data['class'].apply(replace_class_names)


# train_data['class'] = train_data['class'].apply(lambda x: 'attack' if x != 'normal' else x)
# test_data['class'] = test_data['class'].apply(lambda x: 'attack' if x != 'normal' else x)

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

# sampling_strategy = {class_label: 50 for class_label, count in Counter(y_train).items() if count < 5}  # Ensuring at least 50 samples for the smallest classes
# over = RandomOverSampler(sampling_strategy=sampling_strategy)
# under = RandomUnderSampler(sampling_strategy={11: 5000, 9: 5000})  # Reduce counts of largest classes

# pipeline = Pipeline([
#     ('o', over),
#     ('u', under)
# ])

# X_train, y_train = pipeline.fit_resample(X_train, y_train)


# def build_model(input_dim, num_attack_classes, learning_rate):
#     # Define model inputs
#     input_layer = Input(shape=(input_dim,))
    
#     # Shared layers
#     shared_layers = Dense(128, activation='relu')(input_layer)
#     shared_layers = Dense(64, activation='relu')(shared_layers)
#     shared_layers = Dense(32, activation='relu')(shared_layers)
    
#     # Output layer for attack type prediction
#     attack_output = Dense(num_attack_classes, activation='softmax', name='attack_output')(shared_layers)
    

#     # Define the model
#     model = Model(inputs=input_layer, outputs=[attack_output])
    
#     # Compile the model
#     model.compile(optimizer=Adam(learning_rate=learning_rate),
#                   loss={'attack_output': 'sparse_categorical_crossentropy'},
#                   metrics={'attack_output': 'accuracy'})
    
#     return model


def build_model(input_dim,num_classes):
    model = Sequential()

    model.add(Dense(1024, input_dim=input_dim, activation='relu'))  # Input layer to 1st fully connected layer
    # model.add(Dense(1024, activation=None))  # 1st to 2nd layer fully connected with 1024 units
    model.add(BatchNormalization())  # Batch normalization after 1st layer
    model.add(Dense(1024, activation='relu'))  # 2nd to 3rd layer dropout with 1024 units
    model.add(Dropout(0.01))  # Dropout after 2nd layer
    model.add(Dense(768, activation=None))  # 3rd to 4th layer fully connected with 768 units
    model.add(Dropout(0.01))
    model.add(Dense(768, activation=None))  # 4th to 5th layer batch normalization with 768 units
    model.add(BatchNormalization())  # Batch normalization after 4th layer
    model.add(Dense(num_classes, activation='sigmoid'))  # 5th to output layer of 1 unit

    optimizer= Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = len(np.unique(y_encoded))



# You also need to pass the number of features (input_dim) your model should expect
input_dim = X_train.shape[1]

# Now build the model
model = build_model(input_dim,num_classes)

# Display the model's architecture
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping]
                    )

# print(history.history.keys())

model_summary_str = []
model.summary(print_fn=lambda x: model_summary_str.append(x))

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

train_accuracy=np.mean(train_accuracy)
val_accuracy=np.mean(val_accuracy)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")


# Make predictions with your model
predictions = np.argmax(model.predict(X_test), axis=-1)
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
dos_attack_indices = [index for index, label in enumerate(actual_class_labels) if label in dos_attacks]
r2l_attacks_indices = [index for index, label in enumerate(actual_class_labels) if label in r2l_attacks]
u2r_attacks_indices = [index for index, label in enumerate(actual_class_labels) if label in u2r_attacks]
probe_attacks_indices = [index for index, label in enumerate(actual_class_labels) if label in probe_attacks]
normal_indices = [index for index, label in enumerate(actual_class_labels) if label == 'normal']

# Count the number of correct predictions for DoS attack instances
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

# Evaluate the model on the test data
test_results = model.evaluate(X_test, y_test_encoded, verbose=1)
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

fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='Training Accuracy')
ax.plot(history.history['val_accuracy'], label='Validation Accuracy')

# Set labels and title
ax.set_title('Training and Validation Accuracy')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.legend()

experiment_duration = experiment_end_time - experiment_start_time

start_time_formatted = experiment_start_time.strftime("%Y-%m-%d-%H-%M-%S")


file_name = f"expr-{start_time_formatted}"

# Show plot
plt.show()

fig.savefig(f"plot-{file_name}")



with open(f"{file_name}.txt", "w") as file:
    file.write(f"Experiment Details:\n")
    file.write(f"OverSampling\n")
    file.write(f"Start Time: {start_time_formatted}\n")
    file.write(f"Learning Rate: {learning_rate}\n")
    file.write(f"Epochs: {epochs}\n")
    file.write(f"Pateince: {patience}\n")
    file.write(f"Batch Size: {batch_size}\n")
    file.write(f"Val_accuracy: {val_accuracy}\n")
    file.write(f"Accuracy: {test_results[1]}\n")
    file.write(f"DOS Accuracy: {dos_attack_accuracy}\n")
    file.write(f"R2L Accuracy: {r2l_attack_accuracy}\n")
    file.write(f"U2R Accuracy: {u2r_attack_accuracy}\n")
    file.write(f"PROBE Accuracy: {probe_attack_accuracy}\n")
    file.write(f"Normal Accuracy: {normal_accuracy}\n")
    file.write(f"Total Runtime: {experiment_duration}\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("Model Summary:\n")
    file.write("\n".join(model_summary_str))
   
print(f"Experiment details saved to {file_name}")
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

experiment_start_time = datetime.datetime.now()

learning_rate=0.1
epochs=500
batch_size=64
h_layer=1024

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        

column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class"]

# Load data
data = pd.read_csv('kddcup.data.corrected', names=column_names, header=None)
# data['class'] = data['class'].apply(lambda x: 'attack' if x != 'normal' else x)

# Separate features and target
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode target
label_binarizer = LabelBinarizer()
y_encoded = label_binarizer.fit_transform(y)

# Encode categorical features
encoder = LabelEncoder()
x[:, 1] = encoder.fit_transform(x[:, 1])
x[:, 2] = encoder.fit_transform(x[:, 2])
x[:, 3] = encoder.fit_transform(x[:, 3])

normalizer = Normalizer(norm='l2')

# Fit and transform the training data
x = normalizer.fit_transform(x)


# Assuming 'X' is your feature matrix and 'y' is your target variable
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, stratify=y, random_state=42)

# KFold configuration

# kf = KFold(n_splits=n_splits)

# # Define model building function
# def build_model(input_dim):
#     model = Sequential()
#     model.add(Dense(h_layer, input_dim=input_dim, activation='relu'))
#     # model.add(Dense(32, activation='relu'))
#     model.add(Dense(y_encoded.shape[1], activation='sigmoid'))  # Adjust the output layer based on the number of classes
#     optimizer= Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model


def build_model(input_dim):
    model = Sequential()

    model.add(Dense(1024, input_dim=input_dim, activation='relu'))  # Input layer to 1st fully connected layer
    # model.add(Dense(1024, activation=None))  # 1st to 2nd layer fully connected with 1024 units
    model.add(BatchNormalization())  # Batch normalization after 1st layer
    # model.add(Dense(1024, activation='relu'))  # 2nd to 3rd layer dropout with 1024 units
    model.add(Dropout(0.01))  # Dropout after 2nd layer
    # model.add(Dense(768, activation=None))  # 3rd to 4th layer fully connected with 768 units
    model.add(Dense(768, activation=None))  # 4th to 5th layer batch normalization with 768 units
    model.add(BatchNormalization())  # Batch normalization after 4th layer
    model.add(Dense(y_encoded.shape[1], activation='softmax'))  # 5th to output layer of 1 unit

    optimizer= Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize list to store accuracies
# accuracies = []

# KFold Cross-Validation
# for train_index, test_index in kf.split(x):
# x_train, x_test = x[train_index], x[test_index]
# y_train, y_test = y_encoded[train_index], y_encoded[test_index]

# Convert x_train and x_test to float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Build model
model = build_model(x_train.shape[1])

model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,validation_data=(x_test, y_test),callbacks=[early_stopping])
print(history.history.keys())
print("Testing......")
# Evaluate the model
_, accuracy = model.evaluate(x_test, y_test,verbose=1)
print("Accuracy:", accuracy)

# accuracies.append(accuracy)

# Calculate and print the mean accuracy across all folds
# mean_accuracy = np.mean(accuracies)

model_summary_str = []
model.summary(print_fn=lambda x: model_summary_str.append(x))

experiment_end_time = datetime.datetime.now()

# Analyze learning progress
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Find epochs required for learning patterns and detecting attacks
epochs_to_learn = None
overfitting_epochs = None

# Track training accuracy changes over epochs
for epoch in range(1, epochs):  # Assuming epochs is defined earlier in your code
    if epoch > 0 and train_accuracy[epoch] > train_accuracy[epoch - 1]:
        epochs_to_learn = epoch + 1
        break

# Identify overfitting
for epoch in range(1, epochs):
    if epoch > 0 and val_accuracy[epoch] < val_accuracy[epoch - 1]:
        overfitting_epochs = epoch
        break


# Display results
print(f"Epochs required to learn patterns and detect attacks: {epochs_to_learn}")
print(f"Epochs until overfitting occurs: {overfitting_epochs}")


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


file_name = f"experiment-{start_time_formatted}.txt"

fig.savefig(f"plot_{file_name}.png")



with open(file_name, "w") as file:
    file.write(f"Experiment Details:\n")
    file.write(f"Start Time: {start_time_formatted}\n")
    file.write(f"No. of neurons at HL: {h_layer}\n")
    file.write(f"Learning Rate: {learning_rate}\n")
    file.write(f"Epochs: {epochs}\n")
    file.write(f"Batch Size: {batch_size}\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Total Runtime: {experiment_duration}\n")
    file.write("Model Summary:\n")
    file.write("\n".join(model_summary_str))
    
print(f"Experiment details saved to {file_name}")


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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

learning_rate=0.001
epochs=100
batch_size=64
patience=7
h_l=0

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

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

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
    global h_l
    model = Sequential()

    model.add(Dense(1024, input_dim=input_dim, activation='relu'))
    h_l+=1  # Input layer to 1st fully connected layer
    # model.add(Dense(1024, activation=None))  # 1st to 2nd layer fully connected with 1024 units
    model.add(BatchNormalization()) 
    h_l+=1 # Batch normalization after 1st layer
    # model.add(Dense(1024, activation='relu'))  # 2nd to 3rd layer dropout with 1024 units
    model.add(Dropout(0.01))
    h_l+=1  # Dropout after 2nd layer
    model.add(Dense(768, activation=None)) 
    h_l+=1 # 3rd to 4th layer fully connected with 768 units
    model.add(BatchNormalization())
    h_l+=1  # Batch normalization after 4th layer
    model.add(Dense(768, activation=None))  
    h_l+=1 
    model.add(Dense(1, activation='sigmoid'))  # 5th to output layer of 1 unit

    optimizer= Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

num_classes = len(np.unique(y_encoded))



# You also need to pass the number of features (input_dim) your model should expect
input_dim = X_train.shape[1]

# Now build the model
model = build_model(input_dim,num_classes)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)

class_weight_dict[0]=1.5
class_weight_dict[1]=0.8
# Display the model's architecture
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val), class_weight=class_weight_dict,
                    epochs=epochs, batch_size=batch_size, verbose=1,callbacks=[early_stopping])

# print(history.history.keys())

model_summary_str = []
model.summary(print_fn=lambda x: model_summary_str.append(x))

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

train_accuracy=np.mean(train_accuracy)
val_accuracy=np.mean(val_accuracy)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

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
predictions =(model.predict(X_test) > 0.5).astype(int) # np.argmax(model.predict(X_test), axis=-1)

cm = confusion_matrix(y_test_encoded, predictions) #attack=negative, normal=positive
print("Confusion Matrix:\n", cm)
fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=attack_encoder.classes_, yticklabels=attack_encoder.classes_)
ax1.set_xlabel('Predicted labels')
ax1.set_ylabel('True labels')
ax1.set_title('Confusion Matrix')
ax1.xaxis.set_ticklabels(attack_encoder.classes_)
ax1.yaxis.set_ticklabels(attack_encoder.classes_)


file_name = f"expr-{learning_rate}-{epochs}-{batch_size}-{patience}-{h_l}"

fig1.savefig(f"cm-{file_name}.png",dpi=300)

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

adjusted_target_names = [name for i, name in enumerate(attack_encoder.classes_) if i in np.unique(y_test_encoded)]
report = classification_report(y_test_encoded,predictions, target_names=adjusted_target_names, labels=np.unique(y_test_encoded), zero_division=0, output_dict=False)
print(report)

predicted_class_labels = attack_encoder.inverse_transform(predictions)
actual_class_labels = attack_encoder.inverse_transform(y_test_encoded)
print(actual_class_labels)
print(predicted_class_labels)
normal_indices = [index for index, label in enumerate(actual_class_labels) if label == 'normal']
attack_indices = [index for index, label in enumerate(actual_class_labels) if label != 'normal']
# print(normal_indices)
correct_normal_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in normal_indices)
correct_attack_predictions = sum(actual_class_labels[i] == predicted_class_labels[i] for i in attack_indices)

total_normal = len(normal_indices)
total_attack = len(attack_indices)

print(total_normal, total_attack)
print(correct_normal_predictions, correct_attack_predictions)

normal_accuracy = correct_normal_predictions / total_normal if total_normal > 0 else 0
attack_accuracy = correct_attack_predictions / total_attack if total_attack > 0 else 0

print(f"Accuracy for Normal: {normal_accuracy}")
print(f"Accuracy for Attack: {attack_accuracy}")

# Evaluate the model on the test data
test_results = model.evaluate(X_test, y_test_encoded, verbose=1)
# print(test_results)
# print(f"Test Loss: {test_results[0]}")
print(f"Test Accuracy: {test_results[1]}")

experiment_end_time = datetime.datetime.now()

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

# Show plot
# plt.show()

fig.savefig(f"plot-{file_name}.png")



with open(f"{file_name}.txt", "w") as file:
    file.write(f"Experiment Details:\n")
    file.write(f"Start Time: {start_time_formatted}\n")
    file.write(f"Learning Rate: {learning_rate}\n")
    file.write(f"Epochs: {epochs}\n")
    file.write(f"Patience: {patience}\n")
    file.write(f"Batch Size: {batch_size}\n")
    file.write(f"hidden layers: {h_l}\n")
    file.write(f"Val_accuracy: {val_accuracy}\n")
    file.write(f"Accuracy: {test_results[1]}\n")
    file.write(f"Normal accuracy: {normal_accuracy}\n")
    file.write(f"Attack Accuracy: {attack_accuracy} \n")
    file.write(f"Total Runtime: {experiment_duration}\n")
    file.write("Confusion Matrix:\n")
    file.write(str(cm))
    file.write("\nFPR and FNR by Class:\n")
    for label, rates in class_fpr_fnr.items():
        file.write(f"Class: {label}, FPR: {rates['FPR']:.4f}, FNR: {rates['FNR']:.4f}\n")
    file.write("\nClassification Report:\n")
    file.write(report)
    file.write("Model Summary:\n")
    file.write("\n".join(model_summary_str))
    
print(f"Experiment details saved to {file_name}")


# File paths
text_file_path = f"{file_name}.txt"
confusion_matrix_image_path = f"cm-{file_name}.png"
graph_image_path = f"plot-{file_name}.png"
pdf_path = f"pdf-{file_name}+.pdf"

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

# Check space for the confusion matrix and potentially add to new page
if y_position < 350:  # Ensure there's enough space for the confusion matrix
    c.showPage()
    y_position = height - 60

# Add the confusion matrix image
try:
    confusion_matrix = ImageReader(confusion_matrix_image_path)
    c.drawImage(confusion_matrix, 40, y_position - 350, width=520, height=300, preserveAspectRatio=True)
    y_position -= 350  # Adjust y_position after the confusion matrix
except Exception as e:
    print("Failed to load confusion matrix image:", e)

# Check space for the graph and potentially add to new page
if y_position < 350:
    c.showPage()
    y_position = height - 60

# Add the graph image
try:
    graph_image = ImageReader(graph_image_path)
    c.drawImage(graph_image, 40, y_position - 350, width=520, height=300, preserveAspectRatio=True)
    y_position -= 350  # Adjust y_position after the graph
except Exception as e:
    print("Failed to load graph image:", e)

c.save()
print("PDF created at:", pdf_path)
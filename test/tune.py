import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt


tf.config.threading.set_intra_op_parallelism_threads(num_threads=1)
tf.config.threading.set_inter_op_parallelism_threads(num_threads=2)


# Load your data here
train_file_path = 'NSL-KDDTrain+.txt'
test_file_path = 'NSL-KDDTest+.txt'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Assuming the last column is difficulty and the second last is attack type
X = train_data.iloc[:, :-2].values
encoder = LabelEncoder()
X[:, 1] = encoder.fit_transform(X[:, 1])
X[:, 2] = encoder.fit_transform(X[:, 2])
X[:, 3] = encoder.fit_transform(X[:, 3])
y_attack = train_data.iloc[:, -2]
y_difficulty = train_data.iloc[:, -1]

# Preprocess the test set similarly
X_test = test_data.iloc[:, :-2].values
X_test[:, 1] = encoder.fit_transform(X_test[:, 1])
X_test[:, 2] = encoder.fit_transform(X_test[:, 2])
X_test[:, 3] = encoder.fit_transform(X_test[:, 3])
y_attack_test = test_data.iloc[:, -2]
y_difficulty_test = test_data.iloc[:, -1]

# Encode the labels
attack_encoder = LabelEncoder()
y_attack_encoded = attack_encoder.fit_transform(y_attack)

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

y_attack_test_encoded = transform_with_unknown_handling(attack_encoder, y_attack_test)

# No need to encode difficulty if it's already numeric, but ensure it's the correct type
y_difficulty = y_difficulty.astype(int)
y_difficulty_test = y_difficulty_test.astype(int)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split the training data to create a validation set
X_train, X_val, y_attack_train, y_attack_val, y_difficulty_train, y_difficulty_val = train_test_split(
    X_scaled, y_attack_encoded, y_difficulty, test_size=0.2, random_state=42)


# Define the model building function for Keras Tuner
def build_model(hp):
    input_dim = X_train.shape[1]
    num_attack_classes = len(np.unique(y_attack_train))
    num_difficulty_levels = len(np.unique(y_difficulty_train))
    
    inputs = Input(shape=(input_dim,))
    x = inputs
    
    # Tuning the number of units in the Dense layers
    for i in range(hp.Int('num_layers', 1, 3)):
        x = Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu')(x)
        if hp.Boolean('batch_norm_' + str(i)):
            x = BatchNormalization()(x)
    
    attack_output = Dense(num_attack_classes, activation='softmax', name='attack_output')(x)
    difficulty_output = Dense(num_difficulty_levels, activation='softmax', name='difficulty_output')(x)
    
    model = Model(inputs, [attack_output, difficulty_output])
    
    # Tuning the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1,1e-2, 1e-3])
    
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss={'attack_output': 'sparse_categorical_crossentropy', 'difficulty_output': 'sparse_categorical_crossentropy'},
                  metrics={'attack_output': 'accuracy', 'difficulty_output': 'accuracy'})
    return model
objective=kt.Objective("val_attack_output_accuracy", direction="max")
# Instantiate the tuner
tuner = kt.Hyperband(build_model,
                     objective=objective,
                     max_epochs=10,
                     factor=3,
                     directory='nsl_kdd_tuning-2',
                     project_name='dnn_tuning-2')

# Early stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Start search
tuner.search(X_train, {'attack_output': y_attack_train, 'difficulty_output': y_difficulty_train},
             epochs=50,
             validation_split=0.2,
             callbacks=[stop_early],
             verbose=2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Optimal hyperparameter values:")
num_layers = best_hps.get('num_layers')
for i in range(num_layers):
    print(f" - units_{i}: {best_hps.get(f'units_{i}')}")
    if best_hps.get(f'batch_norm_{i}'):
        print(f"   - Batch normalization applied after layer {i}")


# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, {'attack_output': y_attack_train, 'difficulty_output': y_difficulty_train},
                    epochs=50,
                    validation_split=0.2,
                    callbacks=[stop_early],
                    verbose=2)

test_results = model.evaluate(X_test_scaled, [y_attack_test_encoded, y_difficulty_test], verbose=2)
print(test_results)
print(f"Test Loss: {test_results[0]}")
print(f"Attack Type Test Accuracy: {test_results[3]}")
print(f"Difficulty Level Test Accuracy: {test_results[4]}")

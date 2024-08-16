from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Assuming the data is loaded into a DataFrame 'df' and 'column_names' is defined as provided
# For the sake of demonstration, let's create a dummy 'df' DataFrame
column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty"]

# Generate a dummy dataset
# np.random.seed(42)
train_file_path = 'NSL-KDDTrain+.txt'
test_file_path = 'NSL-KDDTest+.txt'

# Assuming your dataset has headers. If not, you might need to specify `header=None` and manually set the column names.
train_data = pd.read_csv(train_file_path, names=column_names, header=None)
test_data = pd.read_csv(test_file_path, names=column_names, header=None)


# Encode categorical features and class labels
encoder = LabelEncoder()
for col in ['protocol_type', 'service', 'flag', 'class']:
    train_data[col] = encoder.fit_transform(train_data[col])
    test_data[col] = encoder.fit_transform(test_data[col])

# Separate features and target
X_train = train_data.drop(['class', 'difficulty'], axis=1)
y_train = train_data['class']

X_test = test_data.drop(['class', 'difficulty'], axis=1)
y_test = test_data['class']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Algorithm implementation
S = X_train.columns.tolist()
best_accuracy = 0
selected_features = []

for i in range(len(S), 0, -1):
    # Create a set of features Si
    Si = S[:i]
    
    # Build a classifier Mi using Si and find the classification accuracy
    Mi = RandomForestClassifier(random_state=42)
    Mi.fit(X_train[Si], y_train)
    y_pred = Mi.predict(X_test[Si])
    accuracy = accuracy_score(y_test, y_pred)
    
    # If classification accuracy Mi ≥ M∗ of S then update S and best_accuracy
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        selected_features = Si
    # else if condition based on threshold is skipped because it's not clear from the algorithm

print(f"Selected features: {selected_features}")
print(f"Best accuracy: {best_accuracy}")

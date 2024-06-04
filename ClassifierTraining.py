import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
print(data_dict.keys())

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Convert labels to integers if they are not already
try:
    labels = labels.astype(int)
except ValueError:
    print("Labels are not numeric. Please ensure labels are numeric for proper sorting.")

# Reassign labels to ensure consecutive ranges
unique_labels = np.unique(labels)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
labels = np.array([label_mapping[label] for label in labels])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'labels': labels}, f)

# Visualize each class
# for label in unique_labels:
#     # Find indices of data points for the current class
#     class_indices = np.where(labels == label)[0]
    
#     plt.figure(figsize=(10, 10))
#     plt.title(f'Hand Landmarks for Class: {label}')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
    
#     # Plot each landmark point
#     for idx in class_indices:
#         landmarks = data[idx]
#         x_coords = landmarks[::2]
#         y_coords = landmarks[1::2]
        
#         plt.scatter(x_coords, y_coords, label=f'{label} Sample {idx}', alpha=0.5)
    
#     plt.gca().invert_yaxis()
#     plt.legend()
#     plt.show()
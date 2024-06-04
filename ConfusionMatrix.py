import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the test data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

x_test = data_dict['data']
y_test = data_dict['labels']

# Load the trained model
with open('model.p', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']

# Make predictions on the test data
y_pred = model.predict(x_test)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
labels = np.unique(y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load your dataset
# Replace 'your_data.csv' with the actual path to your dataset
data = pd.read_csv('creditcard.csv')
print(data.head())
print(data.tail())
# Assuming 'Class' is the target variable
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

# EDA: Visualize patterns and anomalies
plt.figure(figsize=(10, 6))
sns.countplot(x='Class', data=data)
plt.title('Distribution of Classes')
plt.grid(True)

# Preprocessing: Handle missing values (if any)
X.fillna(X.mean(), inplace=True)

# Dimensionality reduction using PCA
pca = PCA(n_components=10)  # You can adjust the number of components based on your requirements
X_pca = pca.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Instantiate the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display evaluation metrics
print(f"Accuracy: {accuracy}  The model correctly classified approximately {accuracy * 100:.1f}% of all transactions.")
print(f"Precision: {precision} Out of all transactions predicted as fraudulent, {precision * 100:.1f}% were actually fraudulent. This measures the accuracy of the positive predictions.")
print(f"Recall: {recall} The model identified approximately {recall * 100:.1f}% of all actual fraudulent transactions. This measures the ability of the model to capture all instances of fraud.")
print(f"F1 Score: {f1} The harmonic mean of precision and recall, providing a balance between the two. It is approximately {f1 * 100:.1f}%.")
print('/n')
print("Confusion Matrix:")
confusion_df = pd.DataFrame(conf_matrix, columns=['Predicted Not Fraud', 'Predicted Fraud'], index=['Actual Not Fraud', 'Actual Fraud'])
print(confusion_df)

plt.show()
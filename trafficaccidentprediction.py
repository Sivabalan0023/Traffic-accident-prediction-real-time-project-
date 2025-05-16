import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv(r"C:\Users\thenn\OneDrive\Desktop\dataset_traffic_accident_prediction1.csv")
print("Initial data shape:", data.shape)

# Show missing values

# Fill missing values with forward fill
data=data.dropna()

# Apply Label Encoding to categorical columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':  # check if column is categorical
        data[col] = le.fit_transform(data[col])
        
print(data.isnull().sum())

# Separate features and label
x = data.drop('Accident', axis=1)
data['Accident']=data['Accident'].astype('int')
y = data['Accident']

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))


def func(sample):
    sample = pd.DataFrame([sample])


    # Encode sample data
    for col in sample.columns:
        if sample[col].dtype == 'object':
            sample[col] = le.fit_transform(sample[col])  # WARNING: overwrites original LabelEncoder — fix below

    # FIX: Use original encoders used during training. Better approach:
    # Save label encoders for each column during training for correct mapping

    # Scale the sample
    sample_scaled = scaler.transform(sample)

    # Predict
    prediction = model.predict(sample_scaled)
    print("Predicted output (0: No Accident, 1: Accident):", prediction[0])
    
    if prediction[0]==1:

        return "accident"
    
    else:
        return "No accident"

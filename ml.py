import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('train.csv')

# Convert assets and liabilities to numerical values
def convert_assets(value):
    if 'Crore' in value:
        return float(value.split()[0]) * 100
    elif 'Lac' in value:
        return float(value.split()[0])
    else:
        return 0

data['Total Assets'] = data['Total Assets'].apply(convert_assets)
data['Liabilities'] = data['Liabilities'].apply(convert_assets)


# Calculate the average criminal records and wealth
avg_criminal_records = data['Criminal Case'].mean()
avg_wealth = (data['Total Assets'] - data['Liabilities']).mean()

# Filter candidates with criminal records and wealth more than the average
criminal_records_above_avg = data[data['Criminal Case'] > avg_criminal_records]
wealth_above_avg = data[(data['Total Assets'] - data['Liabilities']) > avg_wealth]

# Calculate the percentage distribution of parties with candidates having criminal records more than the average
criminal_records_party_dist = criminal_records_above_avg['Party'].value_counts(normalize=True) * 100

# Calculate the percentage distribution of parties with wealthy candidates having wealth more than the average
wealthy_party_dist = wealth_above_avg['Party'].value_counts(normalize=True) * 100

# Plotting the distributions
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].bar(criminal_records_party_dist.index, criminal_records_party_dist.values)
axs[0].set_title('Percentage Distribution of Parties with Candidates\nHaving Criminal Records More Than Average')
axs[0].set_xlabel('Party')
axs[0].set_ylabel('Percentage')

axs[1].bar(wealthy_party_dist.index, wealthy_party_dist.values)
axs[1].set_title('Percentage Distribution of Parties with Wealthy Candidates\nHaving Wealth More Than Average')
axs[1].set_xlabel('Party')
axs[1].set_ylabel('Percentage')

plt.tight_layout()
plt.show()


# Remove unnecessary columns
data = data.drop(columns=['ID', 'Candidate', 'Constituency ∇'])

# Handle missing values if any
data = data.dropna()

# Encode categorical variables
label_encoders = {}
for col in ['Party', 'state']:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])



# Convert education to numerical values
education_mapping = {
    'Others': 1,
    '5th Pass': 2,
    '8th Pass': 3,
    '10th Pass': 4,
    'Literate': 5,
    '12th Pass': 6,
    'Graduate': 7,
    'Graduate Professional': 8,
    'Post Graduate': 9,
    'Doctorate': 10
}

data['Education'] = data['Education'].map(education_mapping)

# Normalize or scale numerical features if necessary

# Split the data into training and testing sets
X = data.drop(columns=['Education'])
y = data['Education']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Train an Ordinal Logistic Regression model
clf = HistGradientBoostingClassifier()
clf.fit(X_train, y_train)

# Reverse the education_mapping dictionary
reverse_education_mapping = {v: k for k, v in education_mapping.items()}

# Make predictions
y_pred = clf.predict(X_test)

# Map numerical values to corresponding keys
y_pred_keys = [reverse_education_mapping[val] for val in y_pred]

# Convert numerical labels to corresponding keys for y_true
y_true_keys = [reverse_education_mapping[val] for val in y_test]

# Evaluate the model
print(classification_report(y_true_keys, y_pred_keys))

# Predict the education level of the winners
# For new data, preprocess it similarly and use clf.predict() to get the predictions


# Load the test dataset
test_data = pd.read_csv('test.csv')

# Remove unnecessary columns
test_data = test_data.drop(columns=['ID', 'Candidate', 'Constituency ∇'])

# Handle missing values in the test dataset
test_data = test_data.dropna()

# Encode categorical variables
for col in ['Party', 'state']:
    test_data[col] = label_encoders[col].transform(test_data[col])

# Convert assets and liabilities to numerical values
test_data['Total Assets'] = test_data['Total Assets'].apply(convert_assets)
test_data['Liabilities'] = test_data['Liabilities'].apply(convert_assets)


# Make predictions
test_predictions = clf.predict(test_data)

# Map numerical values to corresponding keys
test_pred_keys = [reverse_education_mapping[val] for val in test_predictions]

# Create a DataFrame with ID and predicted Education
results_df = pd.DataFrame({'ID': range(len(test_data)), 'Education': test_pred_keys})


# Save the results to a new CSV file
results_df.to_csv('predicted_education.csv', index=False)
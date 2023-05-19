#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Step A: Creating Positive and Negative Sets
train = pd.read_parquet('Train.parquet')
positive_set = train[train['Incident'] == 'TARGET DRUG']
prediction_point = '2023-05-19' 
eligibility_threshold = pd.to_datetime(prediction_point) - pd.DateOffset(days=30)
positive_set = positive_set[positive_set['Date'] <= eligibility_threshold]
negative_set = train[train['Incident'] != 'TARGET DRUG']

# Step B: Feature Engineering
positive_features = positive_set.groupby('Patient-Uid').size().reset_index(name='TargetDrugCount')
# Add more feature engineering steps as needed

# Step C: Model Development and Evaluation
train_set = pd.concat([positive_features, negative_set], ignore_index=True)
X_train = train_set.drop(['Patient-Uid', 'Incident'], axis=1)
y_train = (train_set['Incident'] == 'TARGET DRUG').astype(int)

# Train your model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step D: Generating Predictions for Test Data
test = pd.read_parquet('Test.parquet')
X_test = test.groupby('Patient-Uid').size().reset_index(name='TargetDrugCount')
# Apply the same feature engineering steps to X_test as you did for the training data

# Make predictions on the test data
predictions = model.predict(X_test)

# Save predictions to final_submission.csv
submission_df = pd.DataFrame({'Patient-Uid': X_test['Patient-Uid'], 'Predicted': predictions})
submission_df.to_csv('final_submission.csv', index=False)

# Evaluate the model on a validation set
validation = pd.read_parquet('Validation.parquet')
X_val = validation.groupby('Patient-Uid').size().reset_index(name='TargetDrugCount')
# Apply the same feature engineering steps to X_val as you did for the training data
y_val = (validation['Incident'] == 'TARGET DRUG').astype(int)

# Make predictions on the validation set
val_predictions = model.predict(X_val)

# Evaluate the model using F1-Score
f1 = f1_score(y_val, val_predictions)
print('F1-Score:', f1)


# In[ ]:





# In[ ]:





# Network Intrusion Detection Project

This project utilizes machine learning models to detect anomalies and potential intrusions in network traffic data.

## Data Used

[Network Intrusion dataset (CIC-IDS- 2017)](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) 

## Libraries Used

To install: !pip install pandas numpy sklearn matplotlib seaborn

The following Python libraries were used for data processing, machine learning, and visualization:

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

## Data Preprocessing

Data preprocessing involves handling missing values, encoding labels, scaling features, and splitting the data into training and testing sets.

## ML Models Used

K-means was evaluated to identify patterns that could indicate normal vs. anomalous network traffic. Evaluation metrics include accuracy, precision, recall, and F1-score.

A Random Forest classifier was evaluated for classification between normal and anomalous traffic, achieving high accuracy with precision, recall, and F1-score as evaluation metrics.

Isolation Forest was evaluated as an unsupervised anomaly detection model, isolating potential anomalies without relying on labeled data. Evaluation metrics include accuracy, precision, recall, and F1-score.

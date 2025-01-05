
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    roc_curve, roc_auc_score, matthews_corrcoef
)
# train = pd.read_csv('./dataset split/train.csv',index_col=0)
# valid = pd.read_csv('./dataset split/valid.csv',index_col=0)
# test = pd.read_csv('./dataset split/test.csv',index_col=0)
# fulldf = pd.concat([train, valid, test], axis=0)


def validate_splits(data_util):
    # Combine indices of train, valid, and test
    train_indices = set(data_util.train.index)
    valid_indices = set(data_util.valid.index)
    test_indices = set(data_util.test.index)

    # Check if every row in the original DataFrame is in one of the splits
    all_indices = train_indices.union(valid_indices).union(test_indices)
    missing_indices = set(data_util.df.index) - all_indices
    extra_indices = all_indices - set(data_util.df.index)

    if not missing_indices and not extra_indices:
        print("All rows from the original DataFrame are included in one of the splits.")
    else:
        print(f"Missing rows in splits: {missing_indices}")
        print(f"Extra rows in splits: {extra_indices}")

    # Check for overlap between splits
    overlap_train_valid = train_indices.intersection(valid_indices)
    overlap_train_test = train_indices.intersection(test_indices)
    overlap_valid_test = valid_indices.intersection(test_indices)

    if not overlap_train_valid and not overlap_train_test and not overlap_valid_test:
        print("No overlap between train, valid, and test splits. Each row is in exactly one split.")
    else:
        print("Overlap detected between splits!")
        print(f"Train-Valid Overlap: {overlap_train_valid}")
        print(f"Train-Test Overlap: {overlap_train_test}")
        print(f"Valid-Test Overlap: {overlap_valid_test}")

    # Summary
    print(f"\nTotal rows in original DataFrame: {len(data_util.df)}")
    print(f"Rows in train: {len(data_util.train)}")
    print(f"Rows in valid: {len(data_util.valid)}")
    print(f"Rows in test: {len(data_util.test)}")
    print(f"Total rows across splits: {len(train_indices) + len(valid_indices) + len(test_indices)}")

    if len(data_util.df) == len(train_indices) + len(valid_indices) + len(test_indices):
        print("Row counts match. Data is correctly split.")
    else:
        print("Row counts do not match! Data may be missing or duplicated.")

'''
suitable for classic numerical models: binary, and normalized numerical 
suitable for decision trees: raw 
suitable for nns: label_encoded and normalized numerical
'''


class DataUtil:
    def __init__(self, df, cat_le, cat_oe, bins):
        self.df = df
        self.cat_le = cat_le
        self.cat_oe = cat_oe
        self.bins = bins
        self.nums = [x for x in df.columns if x not in cat_le + cat_oe + bins + ['year', 'top20']]

        self.normalized = None
        self.oe = None
        self.le = None

        # Change data types for bins to bool
        for col in bins:
            self.df[col] = self.df[col].astype(bool)

        # Change data types for categorical columns to object
        for col in cat_le + cat_oe:
            self.df[col] = self.df[col].astype(object)

        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df[col]) < 0.5:  # Convert to category if there are repeated values
                df[col] = df[col].astype('category')

        print("cats lb: ", cat_le)
        print("cats oe: ", cat_oe)
        print("bins: ", bins)
        print("nums :", self.nums)

        ########## one-hot encoding
        self.oe = pd.get_dummies(df[cat_oe], columns=cat_oe, prefix=cat_oe)

        ######### label encoding
        self.label_encoders = {}

        self.le = df[cat_le].copy()
        for col in self.cat_le:
            le = LabelEncoder()
            self.le[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        ######### split
        self.test = df[df['year'] >= 2022]

        # Perform a stratified split
        self.train, self.valid = train_test_split(
            df[df['year'] < 2022],  # Use only data before 2022 for training and validation
            test_size=0.08,  # 8% for validation
            stratify=df[df['year']<2022]['top20'],
            random_state=13  # For reproducibility
        )

        # Check the results
        print(f"Training set size: {len(self.train)}")
        print(f"Validation set size: {len(self.valid)}")
        print(f"Test set size: {len(self.test)}")

        ######### normalize
        self.scaler = StandardScaler()
        self.scaler.fit(self.train[self.nums])
        self.normalized_train = self.scaler.transform(self.train[self.nums])
        self.normalized_valid = self.scaler.transform(self.valid[self.nums])
        self.normalized_test = self.scaler.transform(self.test[self.nums])

    def get_classic_numerical(self):
        # Filter the one-hot-encoded DataFrame for the respective splits
        oe_train = self.oe.loc[self.train.index]
        oe_valid = self.oe.loc[self.valid.index]
        oe_test = self.oe.loc[self.test.index]
        
        # Binary + one-hot encoded + normalized numerical
        train = pd.concat([self.train[self.bins], oe_train, pd.DataFrame(self.normalized_train, columns=self.nums, index=self.train.index)], axis=1)
        valid = pd.concat([self.valid[self.bins], oe_valid, pd.DataFrame(self.normalized_valid, columns=self.nums, index=self.valid.index)], axis=1)
        test = pd.concat([self.test[self.bins], oe_test, pd.DataFrame(self.normalized_test, columns=self.nums, index=self.test.index)], axis=1)
    
        return train, valid, test
    def oe_remove_col(self,col):
        self.oe = self.oe.drop(col,axis=1)
        return self.oe.columns

    def get_decision_tree_data(self):
        # Raw data
        train = self.train
        valid = self.valid
        test = self.test
        return train, valid, test

    def get_nns_data(self):
        # Label encoded + one-hot encoded + normalized numerical
        train = pd.concat([self.le.loc[self.train.index], self.oe.loc[self.train.index], pd.DataFrame(self.normalized_train, columns=self.nums)], axis=1)
        valid = pd.concat([self.le.loc[self.valid.index], self.oe.loc[self.valid.index], pd.DataFrame(self.normalized_valid, columns=self.nums)], axis=1)
        test = pd.concat([self.le.loc[self.test.index], self.oe.loc[self.test.index], pd.DataFrame(self.normalized_test, columns=self.nums)], axis=1)
        return train, valid, test

    def num_cat(self):
        r = []
        for col in self.cat:
            r.append(len(self.label_encoders[col].classes_))
        return r



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    roc_curve, roc_auc_score, matthews_corrcoef, classification_report
)

def evaluate_predictions(pred, label):
    """
    Evaluates predictions against true labels, computes metrics, 
    and generates visualizations for Confusion Matrix and ROC Curve.

    Parameters:
        pred (array-like): Array of predicted labels.
        label (array-like): Array of true labels.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, 
              Matthews correlation coefficient (MCC), ROC AUC, and 
              detailed classification metrics for each label.
    """
    # Metrics
    accuracy = accuracy_score(label, pred)
    precision = precision_score(label, pred)
    recall = recall_score(label, pred)
    mcc = matthews_corrcoef(label, pred)
    fpr, tpr, _ = roc_curve(label, pred)
    roc_auc = roc_auc_score(label, pred)

    # Confusion Matrix
    conf_matrix = confusion_matrix(label, pred)

    # Classification Report
    class_report = classification_report(label, pred, output_dict=True)

    # Print Metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"MCC: {mcc:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")

    # Print Detailed Classification Report
    print("\nClassification Report:")
    print(classification_report(label, pred))

    # Plot Confusion Matrix using Seaborn
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    sns.set_context('notebook', font_scale=1.2)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Plot ROC Curve using Seaborn
    sns.lineplot(x=fpr, y=tpr, label=f'ROC curve (area = {roc_auc:.2f})', color="blue")
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--', color="gray")
    sns.set_context('notebook', font_scale=1.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Return Metrics
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "classification_report": class_report
    }


import pandas as pd
from scipy.stats import zscore

def remove_outliers_zscore(df, threshold=3):
    """
    Removes rows from a DataFrame where any numerical column has a z-score greater than the given threshold.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - threshold (float): The z-score threshold to identify outliers (default is 3).

    Returns:
    - pd.DataFrame: A DataFrame with outliers removed.
    """
    # Compute z-scores for all numerical columns
    z_scores = df.select_dtypes(include=['number']).apply(zscore)

    # Keep rows where all z-scores are within the threshold
    filtered_df = df[(z_scores.abs() <= threshold).all(axis=1)]

    return filtered_df

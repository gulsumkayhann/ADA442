import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Data Cleaning", layout="wide")

# Page title
st.title("Data Cleaning")

# Specify the file path
file_path = "bank-additional.csv"

try:
    # Specify column order
    column_order = ["age", "job", "marital", "education", "default", "housing", "loan",
                    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
                    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
                    "cons.conf.idx", "euribor3m", "nr.employed", "y"]

    # Read the file
    data = pd.read_csv(file_path, delimiter=';', names=column_order, skiprows=1)

    # Display initial data
    st.subheader("Initial Data")
    st.write("Shape of the dataset:", data.shape)
    st.dataframe(data.head())

    # Display missing values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Convert categorical columns to category type
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    data[categorical_columns] = data[categorical_columns].astype('category')

    # Visualize numerical columns against 'age'
    reference_column = 'age'
    numerical_columns = ['duration', 'campaign', 'pdays', 'previous',
                         'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                         'euribor3m', 'nr.employed']

    st.subheader("Scatter Plots")
    st.write("Scatter plots of numerical columns against 'age'")
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, column in enumerate(numerical_columns):
        sns.scatterplot(data=data, x=reference_column, y=column, ax=axes[i], color=(16/255, 37/255, 81/255))
        axes[i].set_title(f'{reference_column} vs {column}', fontsize=12, color=(16/255, 37/255, 81/255))
        axes[i].set_xlabel(reference_column, fontsize=10)
        axes[i].set_ylabel(column, fontsize=10)

    for j in range(len(numerical_columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

    # Boxplots for numerical columns
    st.subheader("Boxplots")
    st.write("Distribution of numerical columns")
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, column in enumerate(numerical_columns):
        sns.boxplot(
            data=data,
            y=column,
            ax=axes[i],
            boxprops=dict(facecolor=(16/255, 37/255, 81/255), edgecolor=(16/255, 37/255, 81/255)),
            medianprops=dict(color="black", linewidth=2)
        )
        axes[i].set_title(f'Distribution of {column}', fontsize=12, color=(16/255, 37/255, 81/255))
        axes[i].set_ylabel(column, fontsize=10)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    for j in range(len(numerical_columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

    # Data Transformation
    st.subheader("Data Transformation")
    st.write("Capping values and filtering rows")

    # Cap the values in the 'campaign' and 'duration' columns at the 90th percentile
    for column in ['campaign', 'duration']:
        threshold = data[column].quantile(0.90)
        data[column] = np.where(data[column] > threshold, threshold, data[column])

    # Remove rows where 'age' > 90
    data = data[data['age'] <= 90]

    # Display transformed data
    st.write("Transformed Data:")
    st.dataframe(data.head())

except FileNotFoundError:
    st.error(f"The file '{file_path}' was not found. Please ensure the file exists in the correct location.")
 
# Set Streamlit page configuration
st.set_page_config(page_title="Data Cleaning and Preprocessing", layout="wide")

# Page title
st.title("Data Cleaning and Preprocessing")

# Specify the file path
file_path = "bank-additional.csv"

try:
    # Specify column order
    column_order = ["age", "job", "marital", "education", "default", "housing", "loan",
                    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
                    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
                    "cons.conf.idx", "euribor3m", "nr.employed", "y"]

    # Read the file
    data = pd.read_csv(file_path, delimiter=';', names=column_order, skiprows=1)

    # Display initial data
    st.subheader("Initial Data")
    st.write("Shape of the dataset:", data.shape)
    st.dataframe(data.head())

    # Display missing values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Convert categorical columns to category type
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    data[categorical_columns] = data[categorical_columns].astype('category')

    # Visualize numerical columns against 'age'
    reference_column = 'age'
    numerical_columns = ['duration', 'campaign', 'pdays', 'previous',
                         'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                         'euribor3m', 'nr.employed']

    st.subheader("Scatter Plots")
    st.write("Scatter plots of numerical columns against 'age'")
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, column in enumerate(numerical_columns):
        sns.scatterplot(data=data, x=reference_column, y=column, ax=axes[i], color=(16/255, 37/255, 81/255))
        axes[i].set_title(f'{reference_column} vs {column}', fontsize=12, color=(16/255, 37/255, 81/255))
        axes[i].set_xlabel(reference_column, fontsize=10)
        axes[i].set_ylabel(column, fontsize=10)

    for j in range(len(numerical_columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

    # Boxplots for numerical columns
    st.subheader("Boxplots")
    st.write("Distribution of numerical columns")
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, column in enumerate(numerical_columns):
        sns.boxplot(
            data=data,
            y=column,
            ax=axes[i],
            boxprops=dict(facecolor=(16/255, 37/255, 81/255), edgecolor=(16/255, 37/255, 81/255)),
            medianprops=dict(color="black", linewidth=2)
        )
        axes[i].set_title(f'Distribution of {column}', fontsize=12, color=(16/255, 37/255, 81/255))
        axes[i].set_ylabel(column, fontsize=10)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    for j in range(len(numerical_columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

    # Data Transformation
    st.subheader("Data Transformation")
    st.write("Capping values and filtering rows")

    # Cap the values in the 'campaign' and 'duration' columns at the 90th percentile
    for column in ['campaign', 'duration']:
        threshold = data[column].quantile(0.90)
        data[column] = np.where(data[column] > threshold, threshold, data[column])

    # Remove rows where 'age' > 90
    data = data[data['age'] <= 90]

    # Display transformed data
    st.write("Transformed Data:")
    st.dataframe(data.head())

    # Add a section for Data Preprocessing
    st.subheader("Data Preprocessing")

    # Scale numerical columns
    scaler = MinMaxScaler()
    numerical_cols = ['age', 'duration', 'campaign', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    st.write("Numerical columns scaled using Min-Max Scaler:")
    st.dataframe(data[numerical_cols].head())

    # Create 'contacted_before' feature from 'pdays'
    data['contacted_before'] = data['pdays'].apply(lambda x: 0 if x == 999 else 1)
    data.drop('pdays', axis=1, inplace=True)
    st.write("Added 'contacted_before' column and dropped 'pdays':")
    st.dataframe(data[['contacted_before']].head())

    # One-hot encode categorical columns
    st.write("Performing one-hot encoding on categorical columns...")
    existing_columns = [col for col in categorical_columns if col in data.columns]
    missing_columns = [col for col in categorical_columns if col not in data.columns]

    if missing_columns:
        st.warning(f"The following categorical columns were not found and skipped: {missing_columns}")

    if existing_columns:
        data = pd.get_dummies(data, columns=existing_columns)
        st.success("One-hot encoding completed.")
        st.write("Updated DataFrame after one-hot encoding:")
        st.dataframe(data.head())
    else:
        st.error("No valid categorical columns found for one-hot encoding.")

except FileNotFoundError:
    st.error(f"The file '{file_path}' was not found. Please ensure the file exists in the correct location.")


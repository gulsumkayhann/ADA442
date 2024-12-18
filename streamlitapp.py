import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Main App Content")
st.write("This is the main area of the app.")

# Sidebar with headers
st.sidebar.title("Sidebar Title")
st.sidebar.header("Section 1")
st.sidebar.write("Content for Section 1 goes here.")

st.sidebar.header("Section 2")
st.sidebar.write("Content for Section 2 goes here.")

st.sidebar.header("Section 3")
st.sidebar.write("Content for Section 3 goes here.")
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

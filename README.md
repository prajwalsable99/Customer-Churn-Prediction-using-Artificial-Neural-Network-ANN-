# Customer Churn Prediction using Artificial Neural Network (ANN)

This project implements a machine learning model using an Artificial Neural Network (ANN) to predict customer churn for a telecom company. The dataset contains various customer information, including demographic details, account information, and the target variable (`Exited`) indicating if the customer churned.

![image](https://github.com/user-attachments/assets/274ba7e7-ca92-480c-8457-e5040f58a990)


## Overview

This project predicts customer churn based on various features such as credit score, balance, and whether the customer is an active member of the service. The dataset is preprocessed and split into training and testing datasets. An ANN is built and trained to classify whether a customer will exit the service or not.

## Dataset

The dataset used in this project is from the `Churn_Modelling.csv` file, which contains customer data for a telecom service. The columns in the dataset are as follows:
```
- `CreditScore`: Credit score of the customer.
- `Geography`: The country of the customer (France, Spain, Germany).
- `Gender`: Gender of the customer (Male/Female).
- `Age`: Age of the customer.
- `Tenure`: Number of years the customer has been with the company.
- `Balance`: Account balance of the customer.
- `NumOfProducts`: Number of products the customer uses.
- `HasCrCard`: Whether the customer has a credit card (1 = Yes, 0 = No).
- `IsActiveMember`: Whether the customer is an active member (1 = Yes, 0 = No).
- `EstimatedSalary`: Estimated salary of the customer.
- `Exited`: Target variable indicating if the customer has exited (1 = Yes, 0 = No).
```

### Requirements
```
- Python 3.x
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
```

### File Structure
```
churn-prediction/
│
├── Churn_Modelling.csv        # Dataset file
├── models/                    # Directory to save trained models
│   ├── genderEncoder.pkl      # Gender encoder
│   ├── oheEncoder.pkl         # One hot encoder
│   ├── scalar.pkl             # StandardScaler
│   └── model.h5               # Trained ANN model
├── ANN-class-impl.ipynb               # Jupyter notebooks (optional)
├── app.py                     # Streamlit app for model inference
```
### Contributing
```
If you'd like to contribute to this project, please fork the repository, create a new branch, and submit a pull request with your changes.
```
### License
```
This project is licensed under the MIT License - see the LICENSE file for details.
```

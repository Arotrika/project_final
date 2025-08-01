# Predictive Maintenance of Industrial Machinery Using Machine Learning

## Overview

The primary goal of this project is to develop a predictive maintenance system that forecasts machine failures in advance using real-time sensor data. By leveraging machine learning, this solution aims to help industries reduce unplanned downtime, optimize maintenance schedules, and minimize operational costs(and is entirely built on IBM Cloud Free-Tire Services).

## Problem Statement

 The objective of this project is to develop a machine learning-based predictive 
maintenance of classification model for a fleet of industrial machines. The model 
should be capable of analyzing real-time sensor data to accurately predict 
impending machine failures before they occur. Specifically, the model must classify 
the type of failure (such as tool wear, heat dissipation issues, or power failure) using 
patterns identified in operational data. 

## Solution Overview

PROPOSED SOLUTION
 The proposed system aims to Predictive maintenance (PdM) leverages real-time sensor data and machine learning models to anticipate machinery failures before they occur, enabling proactive interventions that minimize downtime 
and reduce operational costs. This approach is increasingly feasible and valuable in industrial settings due to advances in IoT, sensor technology, and AI-based analytics. The solution will consist of the following components:
 

**Key Components:**

Data Collection:
 * **Sensors:** Collect data using sensors that monitor parameters such as vibration, temperature, pressure, humidity, current, and acoustic emissions.
 Examples: Infrared thermal sensors (for heat), vibration sensors (for wear or imbalance), and microphones (for acoustic anomalies)
 * **Data Integration:** Integrate data from various systems—sensors, maintenance logs, SCADA, and ERP platforms—into a central database
 * **Data Preprocessing:**
 Clean and preprocess the collected data to handle missing values, outliers, and inconsistencies.
 Feature engineering to extract relevant features from the data that might impact bike demand.
 * **Machine Learning Algorithm:**
 Implement a machine learning algorithm, to train a supervised learning classification model (such as Random Forest) on historical sensor data to learn patterns associated with different machinery failure types.
 Consider Use the trained model to predict the type of impending failure from new real-time sensor inputs, enabling proactive maintenance decisions
 * **Deployment:**
 Real-time sensor data feeds the model, which predicts the likelihood and type of impending failure.
 If the given thresholds are crossed, the model sends alerts through maintenance management systems (CMMS/SCADA), enabling timely, targeted intervention
 Model retraining occurs periodically as new data is collected, ensuring accuracy and adaptability to changing machinery conditions
 * **Evaluation:**
 Assess the model's performance using appropriate metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), or other relevant metrics.
 Cross-validation ensures robustness against overfitting and generalization to unseen data
 * **Result:**
 Stage 1 (Failure Detection): The binary classifier effectively identifies whether a failure will occur, with strong performance metrics shown in the classification report.
 Stage 2 (Failure Type Classification): For actual failures, the model classifies specific failure types with decent accuracy using a multi-class classifier.
 Visualization: The confusion matrix reveals how well the model distinguishes between different failure types, highlighting areas of misclassification
## Data Source
* **Source:** kaggle
* **Link:** [  https://www.kaggle.com/datasets/shivamb/machine
predictive-maintenance-classification ]
* **File in this repository:** `predictive_maintainance.csv`

## Technologies Used

* **IBM Cloud:** The foundational cloud platform.
* **IBM Cloud Object Storage:** For secure storage of the dataset.
* **IBM Watson Studio:** The integrated environment for data science and machine learning workflows.
* **IBM Watson Machine Learning:** Service for managing runtime environments and deploying models.
* **Git & GitHub:** For version control and repository hosting.

## Repository Contents

* `predictive_maintainance.csv`: The raw dataset used for training the model.
* `Predictive_Maintenance_Of_Industrial_Machinary.ipynb`: A Jupyter Notebook,which is inbuilt in IBM Watson Studio's.This notebook provides insights into the preprocessing steps and the model architecture.

## How to Use/Run (Conceptual)

This project was primarily developed and deployed on IBM Cloud. To understand or reproduce the core steps:

1.  **IBM Cloud Account:** Access to an IBM Cloud account (Lite tier is sufficient for this project).
2.  **Provision Services:** Provision IBM Cloud Object Storage, IBM Watson Studio, and IBM Watson Machine Learning services.
3.  **Create Project:** Set up a new project in IBM Watson Studio and link it to your Cloud Object Storage.
4.  **Upload Data:** Upload `predictive_maintainance.csv` to your project's data assets.
5.  **Open Jupyter Notebook**To write the python code so to run the experiment.
6.  **Test Predictions:** The model accurately predicted machine failure types on unseen test data, demonstrating its ability to generalize from training. This confirms its effectiveness for real-time predictive maintenance in industrial settings.

The `Predictive_Maintenance_Of_Industrial_Machinary.ipynb` provides the programmatic details of the generated model and can be run within a Watson Studio notebook environment.

## Model Performance

The Random Forest classifier demonstrated strong performance in predicting machine failure types using sensor data. It achieved an **accuracy** of 92.4% (can be found within the IBM Watson Studio) along with balanced precision and recall across multiple failure categories. The confusion matrix and classification report confirmed that the model generalizes well, making it suitable for real-time predictive maintenance applications.

## Future Enhancements

* **Advanced Algorithms & Optimization:**  •Advanced Algorithms & Optimization:
Future work can explore more sophisticated algorithms such as XGBoost, LightGBM, or deep learning models (e.g., 
neural networks) to improve prediction accuracy, especially for imbalanced or complex datasets. Hyperparameter 
tuning using techniques like Grid Search or Bayesian Optimization could further enhance model performance
* **Real-Time Monitoring & Deployment:** •Real-Time Monitoring & Deployment:
Integrating this predictive maintenance system with IoT-enabled industrial machinery can allow for real-time failure 
detection and automatic alerts. Deploying the model using cloud platforms (AWS, Azure) or edge computing can help 
monitor large-scale operations efficiently
* **Explainability & Trust:** Incorporating tools like SHAP or LIME will make the model's decisions more transparent, enabling engineers to 
understand and trust the predictions, which is especially important in critical maintenance scenarios
* **Integration with Maintenance Scheduling Systems:** By linking the predictive model to automated maintenance planning tools, organizations can move from reactive to 
fully automated preventive maintenance, optimizing downtime, labor, and spare part inventory

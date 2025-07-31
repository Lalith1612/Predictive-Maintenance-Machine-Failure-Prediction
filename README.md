# Predictive Maintenance: Machine Failure Prediction using Deep Learning

<img width="1917" height="885" alt="Screenshot 2025-07-31 152319" src="https://github.com/user-attachments/assets/da9acd6e-0f49-4acc-9033-e956d7406afa" />

# 1. Project Overview

This project presents an end-to-end solution for a critical industrial problem: predicting machine failure. By leveraging real-time sensor data from manufacturing equipment, we can forecast potential breakdowns before they occur, enabling proactive maintenance. This approach, known as **Predictive Maintenance (PdM)**, helps minimize operational downtime, reduce repair costs, and improve overall safety and efficiency.

The core of this project is an **Artificial Neural Network (ANN)** built with TensorFlow and Keras. The model is trained on the "AI4I 2020 Predictive Maintenance Dataset" to classify whether a machine is likely to fail based on its current operational parameters.

The final output is a user-friendly, interactive web application built with **Streamlit**, allowing users to input sensor readings and receive an instant failure prediction probability.

# 2. Tech Stack & Libraries

This project utilizes the following technologies and Python libraries:

- **Core Libraries:** Python 3.9+
- **Data Analysis & Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning & Preprocessing:** Scikit-learn
- **Deep Learning Framework:** TensorFlow, Keras
- **Web Application Framework:** Streamlit
- **Data Ingestion:** Kaggle API
- **Development Environment:** Google Colab (with T4 GPU), Jupyter Notebook

# 4. Methodology

The project follows a standard machine learning lifecycle, from data acquisition to deployment.

# 4.1 Data Ingestion & EDA
- The **AI4I 2020 Predictive Maintenance Dataset** was sourced from Kaggle and ingested directly into the Colab environment using the Kaggle API.
- A thorough **Exploratory Data Analysis (EDA)** was conducted to understand feature distributions, correlations, and the target variable. A key finding was the significant **class imbalance**, with only 3.39% of instances representing a failure. This insight guided the choice of evaluation metrics (Precision, Recall, F1-Score over Accuracy).

# 4.2 Feature Engineering
Based on physical principles and failure modes described in the dataset's documentation (e.g., Heat Dissipation Failure, Power Failure), three new, more informative features were engineered:
- `Temp_Diff`: The difference between process and air temperature, quantifying heat generation.
- `Power`: The calculated mechanical power output (a product of rotational speed and torque).
- `Strain_Index`: An interaction term between tool wear and torque to represent overstrain.

# 4.3 Data Preprocessing
- Data Splitting: The data was split into training (80%) and testing (20%) sets using a **stratified split** to preserve the class distribution.
- Encoding: The categorical `Type` feature was converted to numerical format using One-Hot Encoding.
- Scaling: All numerical features were scaled using `StandardScaler`. The scaler was fitted *only* on the training data to prevent data leakage and then used to transform both the training and testing sets.

# 4.4 Model Development & Evaluation
- ANN Architecture: A sequential ANN was built with Keras, consisting of an input layer, two hidden layers (32 and 16 neurons with ReLU activation), and a single-neuron output layer with a **sigmoid activation function** to produce a probability score.
- Training: The model was trained for 50 epochs using the Adam optimizer and `binary_crossentropy` loss function.
- Evaluation: The ANN's performance was rigorously evaluated on the unseen test set. Due to the class imbalance, **Recall** was prioritized as the key metric, as failing to predict a failure (a False Negative) is far more costly than a false alarm (a False Positive).
- Benchmarking: The ANN was benchmarked against traditional ML models (Logistic Regression, Random Forest, XGBoost) to justify its use and confirm its superior performance on this task.

---

## 5. Streamlit UI Feature Explanation

The Streamlit application provides an intuitive interface for users to get real-time predictions. Each input field in the sidebar corresponds to a key operational parameter of the machine.

| UI Feature | Description | Role in Prediction |

| **Product Type** | A dropdown to select the quality variant of the product being manufactured ('L' for Low, 'M' for Medium, 'H' for High). | This is a categorical feature. Different product types might be associated with different operational stresses, influencing the likelihood of failure. |
| **Air Temperature [K]** | A slider to set the ambient air temperature in Kelvin. | This is a direct sensor reading. It's a baseline for calculating the `Temp_Diff` feature, which is a strong indicator of heat-related failures. |
| **Process Temperature [K]** | A slider to set the temperature of the manufacturing process itself in Kelvin. | A critical sensor reading. High process temperatures, especially relative to air temperature, can signal overheating and potential failure. |
| **Rotational Speed [rpm]** | A slider for the tool's rotational speed in revolutions per minute. | A key operational parameter. Unusually low or high speeds can indicate problems. It is a core component of the engineered `Power` feature. |
| **Torque [Nm]** | A slider for the torque being applied by the tool in Newton-meters. | Represents the rotational force. High torque is a primary indicator of machine strain and is a crucial predictor for overstrain and power-related failures. |
| **Tool Wear [min]** | A slider for the amount of time the tool has been in use, in minutes. | Represents the degradation of the tool. A worn tool requires more torque to operate, increasing the risk of failure. It is a key component of the `Strain_Index` feature. |

When the user clicks the **"Predict Failure"** button, the application takes these six inputs, creates the three engineered features (`Temp_Diff`, `Power`, `Strain_Index`), preprocesses the complete feature set using the saved scaler, and feeds the result to the trained ANN model to generate the final prediction.

# 6. How to Run Locally

To run this application on your local machine, please follow these steps:

Prerequisites:
- Python 3.8 or higher
- `pip` package manager

**Instructions:**

1.  Clone the repository (or download the folder):
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name/deployment
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

4.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

# 7. Future Improvements

- Hyperparameter Tuning: Systematically tune the ANN's hyperparameters (e.g., number of layers, neurons, learning rate) using a tool like Keras Tuner to potentially improve performance.
- Explainable AI (XAI): Integrate SHAP or LIME to provide explanations for each prediction, increasing user trust and providing actionable insights.
- Time-Series Modeling: Adapt the project to predict Remaining Useful Life (RUL) using more complex models like LSTMs or GRUs on a time-series dataset.
- Cloud Deployment: Deploy the application to a permanent cloud service like Streamlit Community Cloud, Heroku, or AWS for persistent public access.

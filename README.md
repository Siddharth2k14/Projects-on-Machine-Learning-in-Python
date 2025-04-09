# 🩺 Diabetes Prediction Model

## 📌 Project Overview
The Diabetes Prediction Model leverages machine learning algorithms to estimate the probability of diabetes in individuals based on medical and demographic features. This project provides a data-driven approach to aid healthcare professionals in early diagnosis and intervention, potentially improving patient outcomes.

## 🧠 Objective
- Predict whether a patient has diabetes using medical parameters.
- Employ various machine learning models to evaluate and compare performance.
- Facilitate early diagnosis and effective healthcare planning.

## 📂 Dataset
- **Source:** Pima Indians Diabetes Dataset
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Target:** Outcome (0: No Diabetes, 1: Diabetes)

## 🛠️ Technologies Used
- Python
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (for modeling and evaluation)

## ⚙️ Data Preprocessing
- **Loading & Exploring:** Checked data types and structure
- **Missing Values:** Confirmed no null values
- **Feature Engineering:** Separated features (X) and target (y)
- **Scaling:** Standardized data using StandardScaler

## 🤖 Machine Learning Models
The following models were trained and tested:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Support Vector Machine (SVM)

## 📊 Evaluation Metrics
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- Mean Squared Error (for comparison insight)

## 📈 Results
Each model was evaluated and compared based on the above metrics to determine the most effective algorithm for diabetes prediction. The best-performing model offers reliable support for early diagnosis.

## 🧾 Conclusion
This predictive model serves as a valuable tool in the healthcare domain, offering early warning signs and enabling informed decision-making. By applying machine learning techniques, the project supports proactive care and optimized resource usage.

## 📚 References
- [Pima Indians Diabetes Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## 👤 Author
**Siddharth Singh**  
B.Tech, Computer Science and Engineering (2024)  
Sikkim Manipal Institute of Technology  
Under supervision of **Mr. Dhruba Ray**

# 📞 Customer Telecom Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-green.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

An end-to-end machine learning project for predicting customer churn in the telecommunications industry using Python, exploratory data analysis, and logistic regression.

## 🌐 Live Demo

**[Try the App →](https://bootcampcustomertelecomchurnproject-greixnnnkmotmtpqr4pit3.streamlit.app/)**

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

Customer churn is a critical metric for telecom companies. This project develops a predictive model to identify customers at risk of leaving, enabling proactive retention strategies. By leveraging machine learning techniques, businesses can reduce customer attrition and improve revenue retention.

### Why Customer Churn Prediction Matters

- **Cost Efficiency**: Acquiring new customers costs 5-25x more than retaining existing ones
- **Revenue Protection**: Identify at-risk customers before they leave
- **Targeted Marketing**: Optimize retention campaigns for high-risk segments
- **Customer Insights**: Understand factors driving customer dissatisfaction

## ✨ Features

- 📊 **Comprehensive EDA**: In-depth exploratory data analysis with visualizations
- 🤖 **Machine Learning Model**: Logistic regression classifier for churn prediction
- 🎨 **Interactive Dashboard**: User-friendly Streamlit web application
- 📈 **Real-time Predictions**: Input customer data and get instant churn probability
- 📉 **Data Preprocessing**: Robust data cleaning and feature engineering pipeline
- 💾 **Model Persistence**: Saved model for quick deployment and inference

## 📁 Project Structure

```
Customer_Telecom_Churn_Project/
│
├── EDA.ipynb                          # Exploratory Data Analysis notebook
├── workspace.ipynb                    # Model development and training notebook
├── app.py                             # Streamlit web application
│
├── TelecomCustomerChurn.csv          # Original dataset
├── cleaned_dataset.csv               # Preprocessed dataset
├── logistic_regression_model.pkl     # Trained model (serialized)
│
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 📊 Dataset

The project uses a telecom customer churn dataset containing:

- **Customer Demographics**: Gender, age, partner status, dependents
- **Account Information**: Tenure, contract type, payment method, billing preferences
- **Service Usage**: Phone service, internet service, online security, technical support, etc.
- **Financial Data**: Monthly charges, total charges
- **Target Variable**: Churn (Yes/No)

### Key Statistics
- Multiple features capturing customer behavior and service subscriptions
- Includes both categorical and numerical variables
- Real-world telecom customer data patterns

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohamed-Tamer-Ai/Customer_Telecom_Churn_Project.git
   cd Customer_Telecom_Churn_Project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Jupyter Notebooks

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook EDA.ipynb
   ```
   Explore data distributions, correlations, and patterns

2. **Model Development**
   ```bash
   jupyter notebook workspace.ipynb
   ```
   View model training, evaluation, and optimization process

### Making Predictions

Through the Streamlit app:
1. Input customer information (demographics, services, account details)
2. Click "Predict Churn"
3. View churn probability and risk classification
4. Get actionable insights for retention strategies

## 🔬 Methodology

### 1. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling and normalization
- Outlier detection and treatment

### 2. Exploratory Data Analysis
- Univariate analysis of all features
- Bivariate analysis with target variable
- Correlation analysis
- Distribution visualization
- Pattern identification

### 3. Feature Engineering
- Creating relevant features from existing data
- Feature selection based on importance
- Dimensionality consideration

### 4. Model Development
- **Algorithm**: Logistic Regression
- **Reason**: Interpretable, efficient, and effective for binary classification
- Train-test split for validation
- Hyperparameter tuning
- Cross-validation

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curve analysis
- Confusion matrix
- Feature importance analysis

### 6. Deployment
- Model serialization using pickle
- Interactive web application with Streamlit
- Cloud deployment for accessibility

## 📈 Model Performance

The logistic regression model achieves:
- **Interpretable Results**: Clear understanding of factors influencing churn
- **Balanced Performance**: Optimized for both precision and recall
- **Practical Application**: Actionable predictions for business decisions

*Note: Specific metrics can be found in the `workspace.ipynb` notebook*

## 🛠️ Technologies Used

### Core Technologies
- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms and tools

### Visualization
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical data visualization

### Machine Learning
- **Logistic Regression**: Classification algorithm
- **Model Serialization**: Pickle for model persistence

### Deployment
- **Streamlit**: Interactive web application framework
- **Streamlit Cloud**: Hosting platform

### Development Tools
- **Jupyter Notebook**: Interactive development environment
- **Git**: Version control

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Try different machine learning algorithms (Random Forest, XGBoost, Neural Networks)
- Enhance feature engineering
- Improve visualization and dashboard
- Add more evaluation metrics
- Optimize model performance
- Expand documentation

## 📝 License

This project is available for educational and research purposes.

## 👤 Contact

**Mohamed Tamer**

- GitHub: [@Mohamed-Tamer-Ai](https://github.com/Mohamed-Tamer-Ai)
- Project Link: [Customer Telecom Churn Project](https://github.com/Mohamed-Tamer-Ai/Customer_Telecom_Churn_Project)
- Live Demo: [Streamlit App](https://bootcampcustomertelecomchurnproject-greixnnnkmotmtpqr4pit3.streamlit.app/)

## 🙏 Acknowledgments

- Dataset sourced from telecom customer churn analysis
- Inspiration from real-world business intelligence needs
- Community support and feedback

---

⭐ **If you find this project helpful, please consider giving it a star!** ⭐

---

### 📚 Additional Resources

- [Logistic Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Customer Churn Analysis Best Practices](https://www.sciencedirect.com/topics/computer-science/customer-churn)

### 🔮 Future Enhancements

- Implement additional ML algorithms for comparison
- Add feature importance visualization
- Create automated reporting system
- Integrate with real-time data sources
- Deploy REST API for predictions
- Add A/B testing framework
- Implement customer segmentation analysis
- Create retention strategy recommendations engine

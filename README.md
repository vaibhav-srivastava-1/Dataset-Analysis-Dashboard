# Dataset Analysis Dashboard

A complete data science web application that automatically analyzes your datasets using various supervised learning algorithms. Built with Streamlit, NumPy, Pandas, Matplotlib, and Seaborn.

## 🚀 Live Application

**Try it now!** Visit the live application:  
👉 **[https://dataset-analysis-dashboard-fu9b3uudzrpqq9pah66iag.streamlit.app/](https://dataset-analysis-dashboard-fu9b3uudzrpqq9pah66iag.streamlit.app/)**

Upload your dataset and get instant analysis with all supervised learning models!

## 🌟 Overview

This project provides an interactive web dashboard that:

- **Accepts your own datasets** (CSV or Excel files)
- **Automatically detects** if your problem is regression or classification
- **Runs all supervised learning algorithms** automatically
- **Displays all visualizations** in an interactive web browser
- **Compares model performance** with detailed metrics

## ✨ Features

### 🔄 Automatic Analysis

- **Smart Problem Detection**: Automatically identifies regression vs classification
- **Data Preprocessing**: Handles missing values and categorical features automatically
- **Feature Engineering**: One-hot encoding for categorical variables
- **Interactive Visualizations**: All plots displayed in the browser

### 📊 Regression Algorithms

- **Linear Regression**: Basic linear relationship modeling
- **Polynomial Regression**: Non-linear relationship modeling (degree=2)
- **Ridge Regression**: L2 regularization for handling multicollinearity
- **Lasso Regression**: L1 regularization for feature selection

### 🎯 Classification Algorithms

- **Logistic Regression**: Binary/multi-class classification using logistic function
- **K-Nearest Neighbors (KNN)**: Instance-based learning (k=5)
- **Support Vector Machine (SVM)**: RBF kernel for non-linear classification
- **Decision Tree**: Tree-based classification with max_depth=5
- **Random Forest**: Ensemble of decision trees (100 estimators)

## 📋 Requirements

- Python 3.8 or higher
- Required packages listed in `requirements.txt`

## 🚀 Installation

1. Clone or download this project
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Start the Web Application

Run the Streamlit app:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Using the Dashboard

1. **Upload Dataset**:

   - Click "Browse files" in the sidebar
   - Upload a CSV or Excel file (.csv, .xlsx, .xls)

2. **Select Target Column**:

   - Choose the column you want to predict from the dropdown

3. **Configure Settings** (Optional):

   - Adjust test size (default: 0.2)
   - Set random state for reproducibility (default: 42)

4. **Run Analysis**:

   - Click the "🚀 Run Analysis" button
   - Wait for all models to train (progress indicators will show)

5. **Explore Results**:
   - View dataset statistics and visualizations
   - See model performance metrics
   - Compare all models side-by-side
   - Explore detailed classification reports

## 📁 Project Structure

```
.
├── app.py                 # Streamlit web application
├── main.py                # Original script (for reference)
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🎨 Dashboard Features

### Data Exploration

- Dataset preview with first 10 rows
- Dataset statistics (rows, columns, missing values)
- Target distribution visualization
- Correlation heatmap (for datasets with ≤20 features)
- Descriptive statistics table

### Model Results

- **Regression**:
  - MSE, MAE, and R² scores for all models
  - Actual vs Predicted scatter plots
  - Model comparison bar chart
- **Classification**:
  - Accuracy scores for all models
  - Confusion matrices for each model
  - Detailed classification reports (precision, recall, F1-score)
  - Model comparison bar chart

## 📚 Key Libraries Used

- **Streamlit**: Interactive web application framework
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Basic plotting and visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning algorithms and utilities
- **OpenPyXL**: Excel file support

## 📊 Model Evaluation Metrics

### Regression Models

- **MSE (Mean Squared Error)**: Average squared difference between actual and predicted values
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Coefficient of determination (higher is better, max=1.0)

### Classification Models

- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of prediction accuracy

## 🎯 Supported File Formats

- **CSV files** (.csv)
- **Excel files** (.xlsx, .xls)

## 🔧 Dataset Requirements

### For Best Results:

- **Numeric features** work best (categorical features are automatically encoded)
- **Target column** should be clearly identifiable
- **No excessive missing values** (automatically handled, but cleaner data is better)
- **At least 50-100 rows** recommended for meaningful results

### Target Column:

- **Regression**: Numeric column with many unique values
- **Classification**: Numeric column with few unique values (≤10) or categorical column

## 💡 Tips

1. **Large Datasets**: For datasets with many features (>20), correlation heatmap may be disabled for performance
2. **Categorical Features**: Automatically handled with one-hot encoding
3. **Missing Values**: Automatically filled (median for numeric, mode for categorical)
4. **Model Training**: All models are trained with standardized features for optimal performance

## 🎉 Example Workflow

1. Upload your dataset (e.g., `house_prices.csv`)
2. Select target column (e.g., `price`)
3. System detects it's a regression problem
4. All 4 regression models train automatically
5. View R² scores, predictions plots, and comparisons
6. Identify the best performing model!

## 🔄 Alternative: Command Line Script

If you prefer the original command-line version:

```bash
python main.py
```

This will generate synthetic datasets and save visualization PNG files.

## 📝 Notes

- All models use standardized features for better performance
- Random seeds are set for reproducibility
- The application automatically handles data preprocessing
- Visualizations are interactive and displayed in the browser
- No need to save files - everything is shown in the dashboard!

## 🐛 Troubleshooting

- **"No module named streamlit"**: Run `pip install streamlit`
- **File upload issues**: Ensure your file is CSV or Excel format
- **Memory errors**: Try with a smaller dataset or reduce test size
- **Slow training**: Large datasets may take time; progress indicators will show status

### 😴 App Goes to Sleep After Inactivity

**If deployed on Streamlit Cloud (Free Tier):**
- Streamlit Cloud free tier apps automatically sleep after ~1 hour of inactivity
- This is a platform limitation and cannot be prevented programmatically
- **Solutions:**
  - Upgrade to Streamlit Cloud paid plan for always-on apps
  - Use external monitoring services (e.g., UptimeRobot) to ping your app every few minutes
  - Deploy to alternative platforms like Heroku, Railway, or Render with always-on options

**If running locally:**
- The app should not sleep, but session timeouts may occur
- A keep-alive mechanism has been added to the code
- The `.streamlit/config.toml` file extends timeout settings
- Keep the browser tab active and interact with the app periodically

## 👨‍💻 About the Developer

**Vaibhav Srivastava**  
MCA AI and ML Student  
Lovely Professional University (LPU)

This project was developed as part of my academic coursework and personal learning in Data Science and Machine Learning. It demonstrates practical implementation of various supervised learning algorithms and their application to real-world datasets.

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

Special thanks to the open-source community for providing excellent libraries like Streamlit, Scikit-learn, Pandas, and NumPy that made this project possible.

# Play Predictor Machine Learning Project
This project is a Machine Learning pipeline for predicting whether to play based on weather conditions and temperature using multiple classification algorithms. It includes data exploration, preprocessing, and the implementation of various classification algorithms such as Decision Tree, K-Nearest Neighbors, Logistic Regression, Support Vector Machine, Random Forest, and Voting Classifier.

## Dataset
The dataset used in this project is PlayPredictor.csv. The dataset includes the following features:

- Whether: The weather condition (e.g., Sunny, Rainy, etc.).
- Temperature: The temperature level (e.g., Hot, Cool, etc.).
- lay: Whether the game is played (Yes/No), which is the target variable.

## Project Structure
- PlayPredictor.csv: The dataset file.
- main.py: The Python script containing the full implementation of the machine learning models.
- README.md: This documentation file.

## Data Preprocessing
- Label Encoding: Categorical variables Whether, Temperature, and Play are converted into numerical form using LabelEncoder.
- Standard Scaling: The features are scaled using StandardScaler to ensure that the models which depend on distances (like KNN and SVM) perform better.

## Exploratory Data Analysis (EDA)
The code performs an in-depth EDA that includes:

- Displaying the structure and summary statistics of the dataset.
- Checking for null values, duplicates, and unique values in each column.
- Visualizing the distribution of features with count plots.
- Generating a correlation matrix heatmap for visualizing feature relationships.

## Machine Learning Models
The following classification models are trained and evaluated:

- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- Voting Classifier: Combines predictions from all the above models for a final ensemble prediction.
  Each model is trained on the dataset and evaluated using accuracy score and a confusion matrix.

## Model Performance
The models' performance is displayed with:

- Accuracy Score for both training and testing datasets.
- Confusion Matrix to evaluate the performance of each classifier visually.

## Conclusion
This project demonstrates a variety of machine learning algorithms and how to use them for classification tasks. The Voting Classifier provides an ensemble solution combining the strengths of multiple algorithms.

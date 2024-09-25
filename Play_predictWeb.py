import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

st.set_page_config(page_title="Play_predict",layout="centered")

# Streamlit app function
def MLALGO():
    # Load data via file uploader in Streamlit
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Dataset overview
        st.write("First Five rows of dataset:", df.head())
        st.write("Size of dataset:", df.shape)
        st.write("Columns in dataset:", df.columns.tolist())
        st.write("Numerical columns:", [col for col in df.columns if df[col].dtypes != "object"])
        st.write("Categorical columns:", [col for col in df.columns if df[col].dtypes == "object"])
        st.write("Column Info:", df.info())
        st.write("Statistical Summary:", df.describe())
        st.write("Duplicate Values:", df.duplicated().sum())
        st.write("Null Values:", df.isnull().sum())
        st.write("Unique values in dataset:", df.nunique())

        # EDA analysis
        st.subheader("Exploratory Data Analysis (EDA)")
        for col in df.columns[1:]:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=df[col])
            plt.title(f"Distribution of {col}")
            st.pyplot(plt)
        
        # Display count plots
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        sns.countplot(x="Whether", data=df, ax=ax[0])
        sns.countplot(x="Temperature", data=df, ax=ax[1])
        sns.countplot(x="Play", data=df, ax=ax[2])
        st.pyplot(fig)
        
         # Data Preprocessing
        le = LabelEncoder()
        df["Whether"] = le.fit_transform(df["Whether"])
        df["Temperature"] = le.fit_transform(df["Temperature"])
        df["Play"] = le.fit_transform(df["Play"])
        
        # Display the correlation matrix
        st.subheader("Correlation Matrix")
        plt.figure(figsize=(8, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)

       
        
        df = df.drop("Unnamed: 0", axis=1)

        x = df.drop("Play", axis=1)
        y = df["Play"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        scale = StandardScaler()
        x_train_scaled = scale.fit_transform(x_train)
        x_test_scaled = scale.transform(x_test)

        # Create classifiers and models
        classifiers = {
            "Decision Tree": DecisionTreeClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Support Vector Machine": SVC(probability=True),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "Voting Classifier": VotingClassifier(estimators=[
                ("DT", DecisionTreeClassifier()),
                ("KNN", KNeighborsClassifier(n_neighbors=5)),
                ("LR", LogisticRegression(max_iter=200)),
                ("SVM", SVC(probability=True)),
                ("RFC", RandomForestClassifier(n_estimators=100))
            ], voting="soft"),
            "Bagging Classifier": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42),
            "Boosting Classifier": AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, algorithm="SAMME", random_state=42)
        }

        # Model Selection
        model_name = st.selectbox("Select Classifier", list(classifiers.keys()))
        model = classifiers[model_name]

        # Training and predictions
        if model_name in ["K-Nearest Neighbors", "Logistic Regression", "Support Vector Machine", "Voting Classifier"]:
            model.fit(x_train_scaled, y_train)
            train_pred = model.predict(x_train_scaled)
            test_pred = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)

        # Display accuracy scores
        st.write(f"Training Accuracy of {model_name} model:", accuracy_score(train_pred, y_train))
        st.write(f"Testing Accuracy of {model_name} model:", accuracy_score(test_pred, y_test))

        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, test_pred), annot=True, cmap="Blues")
        plt.xlabel("Actual values")
        plt.ylabel("Predicted values")
        plt.title(f"Confusion Matrix of {model_name} model")
        st.pyplot(plt)

def main():
    st.title("Machine Learning Algorithm Comparison")
    MLALGO()

if __name__ == "__main__":
    main()
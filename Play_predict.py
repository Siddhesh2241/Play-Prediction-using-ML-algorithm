import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix



def MLALGO():

    # Load data 

    df = pd.read_csv("PlayPredictor.csv")
    
    # Analyse data 
    
    print("First Five rows of dataset is\n",df.head())
    print("\nsize of dataset\n",df.shape)
    print("\nColumns name in the dataset\n",df.columns.tolist())
    print("\nNumrecial columns in dataset\n",[col for col in df.columns if df[col].dtypes != "object" ])
    print("\nCategorical columns in dataset\n",[col for col in df.columns if df[col].dtypes == "object" ])
    print("\nInfo of column \n",df.info())
    print('\nStatistical data of column\n',df.describe())
    print("\nCheck dulicate in dataset\n",df.duplicated())
    print('\nCheck null values\n',df.isnull().sum())
    print("\nCheck unique values in dataset\n",df.nunique())

    # EDA analysis

    for col in df.columns[1:]:
        plt.figure(figsize=(8,6))
        sns.countplot(x=df[col])
        plt.title(f"Ditribution of {col}")
        plt.show()

    
    fig,ax = plt.subplots(1,3,figsize = (15,6)) 
    sns.countplot(x="Whether",data=df,ax=ax[0])
    sns.countplot(x="Temperature",data=df,ax=ax[1]) 
    sns.countplot(x="Play",data=df,ax=ax[2])
    plt.show()

    plt.figure(figsize=(8,6))
    sns.countplot(x="Whether",data=df,hue="Temperature")
    plt.show()

    plt.figure(figsize=(8,6))
    sns.countplot(x="Temperature",data=df,hue="Play")
    plt.show()
    
    # Preprocessing data
    
    # use LabelEncoder

    le = LabelEncoder()
     
    df["Whether"] = le.fit_transform(df["Whether"])
    df["Temperature"] = le.fit_transform(df["Temperature"])
    df["Play"] = le.fit_transform(df["Play"])
    
    
    df = df.drop("Unnamed: 0",axis=1)

    plt.figure(figsize=(8,8))
    sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
    plt.title("Corelation of matrix")
    plt.show()
    
    x = df.drop("Play",axis=1)
    y = df["Play"]
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    # use StandardScaler

    scale = StandardScaler()

    x_train_scaled = scale.fit_transform(x_train)
    x_test_scaled = scale.transform(x_test)

    # Decesion tree

    modelDT = DecisionTreeClassifier()
    modelDT.fit(x_train,y_train)
    train_pred = modelDT.predict(x_train)

    print("\nTraining Accuracy score of Decesion tree model",accuracy_score(train_pred,y_train))

    test_pred = modelDT.predict(x_test)
    print("Testing Accuracy score of Decesion tree model",accuracy_score(test_pred,y_test))

    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test,test_pred),annot=True,cmap="Blues")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Confusion matrix of Decision tree model")
    plt.show()

    
    # Knearest_neigbhor

    modelKNN = KNeighborsClassifier(n_neighbors=5)
    modelKNN.fit(x_train_scaled,y_train)
    train_pred = modelKNN.predict(x_train_scaled)
    
    print("\nTraining Accuracy score of K_nearest_neighbhor model",accuracy_score(train_pred,y_train))

    test_pred = modelKNN.predict(x_test_scaled)
    print("Testing Accuracy score of K_nearest_neigbhor model",accuracy_score(test_pred,y_test))

    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test,test_pred),annot=True,cmap="Blues")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Confusion matrix of Knearest_neigbhor model")
    plt.show()
    
    # logistic regression 
    
    modelLR = LogisticRegression(max_iter=200)
    modelLR.fit(x_train_scaled,y_train)
    train_pred = modelLR.predict(x_train_scaled)

    print("\nTraining Accuracy score of Logistic regression model",accuracy_score(train_pred,y_train))

    test_pred = modelLR.predict(x_test_scaled)
    print("Testing Accuracy score of Logistic regression model",accuracy_score(test_pred,y_test))
    
    
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test,test_pred),annot=True,cmap="Blues")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Confusion matrix of logistic regression model")
    plt.show()

    # Support vector machine

    modelSVM = SVC(probability=True)
    modelSVM.fit(x_train_scaled,y_train)
    train_pred = modelSVM.predict(x_train_scaled)

    print("\nTraining Accuracy score of Support vector machine model",accuracy_score(train_pred,y_train))

    test_pred = modelSVM.predict(x_test_scaled)
    print("Testing Accuracy score of Support vector machine model",accuracy_score(test_pred,y_test))
    
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test,test_pred),annot=True,cmap="Blues")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Confusion matrix of Support vector machine model")
    plt.show()

    # Random forest classifier

    modelRFC = RandomForestClassifier(n_estimators=100)
    modelRFC.fit(x_train,y_train)
    train_pred = modelRFC.predict(x_train)

    print("\nTraining Accuracy score of Random forest classifier model",accuracy_score(train_pred,y_train))

    test_pred = modelRFC.predict(x_test)
    print("Testing Accuracy score of Random forest classifier model",accuracy_score(test_pred,y_test))
    
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test,test_pred),annot=True,cmap="Blues")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Confusion matrix of Random forest classifier model")
    plt.show()

    # voting classifier

    Voting_clf = VotingClassifier(estimators=[
                 ("DT",modelDT),
                 ("KNN",modelKNN),
                 ("LR",modelLR),
                 ("SVM",modelSVM),
                 ("RFC",modelRFC)
            ],voting="soft")
    
    Voting_clf.fit(x_train_scaled,y_train)

    train_pred = Voting_clf.predict(x_train_scaled)

    print("\nTraining Accuracy score of Voting classifier model",accuracy_score(train_pred,y_train))

    test_pred = Voting_clf.predict(x_test_scaled)
    print("Testing Accuracy score of voting classifier model",accuracy_score(test_pred,y_test))
    
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test,test_pred),annot=True,cmap="Blues")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Confusion matrix of voting classifier model")
    plt.show()

    
def main():
   
   MLALGO()

if __name__ == "__main__":
    main()
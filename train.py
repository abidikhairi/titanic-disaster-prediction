import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    data = pd.read_csv("./data/train-v1.csv")

    x_data = data[['Pclass','Sex_Code','Age','SibSp','Parch','Fare','Embarked_Code']]
    y_data = data['Survived']

    # 80% train -> 20% test
    # 100 -> 80 - 20 | 95 - 5
    # 10000 -> 8000 - 2000

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    svm = SVC()

    # train the model here !
    svm_model = svm.fit(x_train, y_train)
    
    y_pred = svm_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f} %")
    print(f"Recall: {recall * 100:.2f} %")
    # Recall = True Positives / (True Positives + False Negatives)
    # 0 -> No, 1 -> Yes

    cm = confusion_matrix(y_test, y_pred)

    for row in cm:
        for col in range(len(row)):
            print(row[col], end="\t")
        print()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def knn(x_train, x_test, y_train, y_test, k, simpleMode=True):
    knn_fit = KNeighborsClassifier(n_neighbors=k)
    knn_fit.fit(x_train, y_train)
    prediction = knn_fit.predict(x_test)
    confusionMatrix = confusion_matrix(y_test, prediction, labels=["Yes, and I tested positive", "Yes, and I tested negative"])
    TP = confusionMatrix[0, 0]
    FP = confusionMatrix[0, 1]
    FN = confusionMatrix[1, 0]
    TN = confusionMatrix[1, 1]

    if simpleMode==True:
        return (TP + TN)/(TP + FP + FN + TN)
    else:
        accuracy = (TP + TN)/(TP + FP + FN + TN)
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        falsePositiveRate = FP/(FP + TN)
        falseNegativeRate = FN/(TP + FN)

        returnList = []
        returnList.append(prediction)                 # output[0] is the prediction results
        returnList.append(confusionMatrix)            # output[1] is the confusion matrix
        returnList.append(accuracy)                   # output[2] is model accuracy
        returnList.append(precision)                  # output[3] is model precision
        returnList.append(recall)                     # output[4] is model recall
        returnList.append(falsePositiveRate)          # output[5] is model falsePositiveRate
        returnList.append(falseNegativeRate)          # output[6] is model falseNegativeRate
        return returnList


if __name__ == "__main__":
    dataset = pd.read_csv("COVID_Data_Cleaned.csv")
    pd.options.display.max_columns = None  # Do not hide column info
    pd.options.display.max_rows = None  # Do not hide row info
    dataset.drop("Year", axis="columns", inplace=True)
    dataset.fillna("Not_Applicable", inplace=True)
    posNegData = dataset.loc[(dataset["i3_health"] == "Yes, and I tested negative") | (dataset["i3_health"] == "Yes, and I tested positive")]
    posNegData = posNegData.reset_index(drop=True)

    # How is target distributed
    posNegData["i3_health"].value_counts().plot(kind="bar", title="COVID")
    plt.xticks(rotation="horizontal")
    plt.show()

    # Correlation matrix for numerical variables
    correlationMatrix = posNegData.corr()

    # Multivariate Analysis
    pairPlot = sns.pairplot(data=posNegData, hue="i3_health")
    plt.show()

    # Bot plots of target with numeric predictors
    # Find all numerical variables
    dataTypes = posNegData.dtypes
    numericalVariableList = []
    n = 0
    for value in dataTypes:
        if value == "int64" or value == "float64":
            numericalVariableList.append(dataTypes.index[n])
        n = n + 1
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharey=False)
    fig.suptitle('Covid Result vs Predictors')
    n = 1
    for name in numericalVariableList:
        x = 0
        y = 0
        if n == 1:
            x = 0
            y = 0
        elif n == 2:
            x = 1
            y = 0
        elif n == 3:
            x = 2
            y = 0
        elif n == 4:
            x = 0
            y = 1
        elif n == 5:
            x = 1
            y = 1
        elif n == 6:
            x = 2
            y = 1
        elif n == 7:
            x = 0
            y = 2
        elif n == 8:
            x = 1
            y = 2
        else:
            x = 2
            y = 2
        n = n + 1
        sns.boxplot(ax=axes[y, x], data=posNegData, y=posNegData[name], x=posNegData["i3_health"])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    plt.show()


    # KNN model
    # First split our data in to training and testing dataset
    posNegData_x = posNegData.drop("i3_health", axis="columns")
    posNegData_y = posNegData["i3_health"]
    posNegData_x = pd.get_dummies(data=posNegData_x, drop_first=True)  # One hot encoding
    x_train, x_test, y_train, y_test = train_test_split(posNegData_x, posNegData_y, test_size=0.2, random_state=1)
    modelTuning = []
    n = 1
    while n <= 100:
        output = knn(x_train, x_test, y_train, y_test, n)
        modelTuning.append([n, output])
        n = n + 1
    modelTuning  # k = 19 yields the best result

    output = knn(x_train, x_test, y_train, y_test, 19, simpleMode=False)
    print("Model accuracy is: " + str(round(output[2], 4)*100) + "%")
    print("Model precision is: " + str(round(output[3], 4)*100) + "%")
    print("Model recall is: " + str(round(output[4], 4)*100) + "%")
    print("Model false positive rate is: " + str(round(output[5], 4)*100) + "%")
    print("Model false negative rate is: " + str(round(output[6], 4)*100) + "%")

























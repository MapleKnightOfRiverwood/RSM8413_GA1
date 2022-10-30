import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

if __name__ == "__main__":  # Main class
    dataset = pd.read_csv("COVID-19BehaviorData_CAN_USA.csv")
    pd.options.display.max_columns = None  # Do not hide column info
    pd.options.display.max_rows = None  # Do not hide row info

    # Dataset preview
    dataset.shape
    dataset.dtypes

    # Data preparation and wrangling
    dataset.shape

    # Check for missing values
    columnWithMissingValue = []
    for columnName in dataset:
        if dataset[columnName].isnull().sum() != 0:
            columnWithMissingValue.append(columnName)
    print(columnWithMissingValue)  # No null missing values

    for columnName in dataset:
        if (dataset[columnName] == " ").sum() != 0:
            columnWithMissingValue.append(columnName)
    print(columnWithMissingValue)  # These columns have missing values in space form

    for columnName in columnWithMissingValue:  # Check percentage of data missing
        percentageMissing = (((dataset[columnName] == " ").sum())/len(dataset[columnName]))*100
        print(columnName + " has " + str(round(percentageMissing, 2)) + "% of data missing.")

    # Replace all missing values with "N/A"
    dataset.replace(" ", "Not_Applicable", inplace=True)

    n = 0
    while n <= len(dataset["RecordNo"]) - 1:
        countryCode = dataset.loc[n, "RecordNo"][0] + dataset.loc[n, "RecordNo"][1] + dataset.loc[n, "RecordNo"][2]
        if countryCode == "CAN":
            dataset.loc[n, "RecordNo"] = "CAN"
        else:
            dataset.loc[n, "RecordNo"] = "USA"
        n = n + 1

    # Check column types for possible inaccuracy error
    print(dataset.dtypes)  # household_size and household_children are string instead of int
    print(dataset["household_size"].unique())  # Summary of unique classes for categorical variable
    print(dataset["household_children"].unique())  # Summary of unique classes for categorical variable
    # We noticed that there are non-numerical values like "Don't know", "Prefer not to say" and "# or more"
    # in household_size and household_children features, there are a lot of _NA_ in i14_health_other, and
    # "RecordNo" column is not useful since it is identical with index column

    # Check percentage of wrong values in i14_health_other
    i14ho_percentageWrong = (((dataset["i14_health_other"] == "__NA__").sum())/len(dataset["i14_health_other"]))*100
    print("i14_health_other has " + str(round(i14ho_percentageWrong, 2)) + "% of data missing.")
    # Since more than 92.42% of data is missing, and it is unstructured, we remove the column
    dataset.drop("i14_health_other", axis="columns", inplace=True)

    # Change "# and more" to "#"
    dataset.loc[dataset["household_size"] == "8 or more", "household_size"] = "8"
    dataset.loc[dataset["household_children"] == "5 or more", "household_children"] = "5"

    # Check percentage of wrong values in household_size and household_children
    hs_hasWrong = np.logical_or(dataset["household_size"] == "Don't know", dataset["household_size"] == "Prefer not to say")
    hs_percentageWrong = ((hs_hasWrong.sum()) / len(dataset["household_size"])) * 100
    print("household_size has " + str(round(hs_percentageWrong, 2)) + "% of data is wrong")
    hc_hasWrong = np.logical_or(dataset["household_children"] == "Don't know", dataset["household_children"] == "Prefer not to say")
    hc_percentageWrong = ((hc_hasWrong.sum()) / len(dataset["household_children"])) * 100
    print("household_children has " + str(round(hc_percentageWrong, 2)) + "% of data is wrong")
    # Since percentage missing is not large, we can try to predict inaccuracy errors in household_size

    # Replace inaccuracy values in household_size
    # Create a new dataset to preserve the original dataset
    hs_dataset = dataset.copy()  # Deep copy rather than pointer
    # Backup a test dataset without hot encoding with all error rows
    hs_testDataset_noDummy = hs_dataset.loc[(hs_dataset["household_size"] == "Don't know") | (hs_dataset["household_size"] == "Prefer not to say")]
    # Backup a training dataset without hot encoding with all error rows
    hs_trainingDataset_noDummy = hs_dataset.loc[(hs_dataset["household_size"] != "Don't know") & (hs_dataset["household_size"] != "Prefer not to say")]

    hs_hsColumnBackup = hs_dataset["household_size"]  # Backup target column
    hs_dataset.drop("household_size", axis="columns", inplace=True)  # Drop target column for one hot encoding
    hs_hcColumnBackup = hs_dataset["household_children"]  # Backup other column with inaccuracy error
    hs_dataset.drop("household_children", axis="columns", inplace=True)  # Remove column with inaccuracy error for one hot encoding
    hs_indexBackup = hs_dataset["Index"]  # Backup index column
    hs_dataset.drop("Index", axis="columns", inplace=True)  # Remove index column for one hot encoding
    hs_timeColumnBackup = hs_dataset["endtime"]  # Backup date time column
    hs_dataset.drop("endtime", axis="columns", inplace=True)  # Remove date time column for one hot encoding
    hs_dataset = pd.get_dummies(data=hs_dataset, drop_first=True)  # One hot encoding
    # Add back backed up columns
    hs_dataset["household_size"] = hs_hsColumnBackup
    hs_dataset["household_children"] = hs_hcColumnBackup
    hs_dataset["Index"] = hs_indexBackup
    hs_dataset["endtime"] = hs_timeColumnBackup

    # Create a test dataset with all error rows
    hs_testDataset = hs_dataset.loc[(hs_dataset["household_size"] == "Don't know") | (hs_dataset["household_size"] == "Prefer not to say")]
    # Create a training dataset with all error rows
    hs_trainingDataset = hs_dataset.loc[(hs_dataset["household_size"] != "Don't know") & (hs_dataset["household_size"] != "Prefer not to say")]
    hs_trainingDataset["household_size"] = hs_trainingDataset["household_size"].astype("int64")  # Convert hs to int
    # Separate features and target
    hs_x_train = hs_trainingDataset.drop(["household_size", "household_children", "Index", "endtime"], axis="columns")
    hs_y_train = hs_trainingDataset["household_size"]
    hs_x_test = hs_testDataset.drop(["household_size", "household_children", "Index", "endtime"], axis="columns")
    # Model Training
    lr = LinearRegression()
    lr.fit(hs_x_train, hs_y_train)
    # Predict hs_y_test
    hs_y_test = lr.predict(hs_x_test)
    max(hs_y_test)  # Make sure the max value is not greater than 8
    min(hs_y_test)  # Make sure the min value is not less than 0
    hs_y_test = hs_y_test.round(decimals=0).astype("int64")  # Round and change type to integer
    hs_testDataset_noDummy["household_size"] = hs_y_test  # Add prediction result to test data set
    # Combine test dataset and training dataset to obtain new dataset
    hs_dataset = pd.concat([hs_trainingDataset_noDummy, hs_testDataset_noDummy])
    hs_dataset.sort_values(["Index"], inplace=True)
    # Replace our main dataset
    dataset = hs_dataset
    dataset["household_size"] = dataset["household_size"].astype("int64")  # Change household_size to int

    # Replace inaccuracy values in household_children
    # Create a new dataset to preserve the original dataset
    hc_dataset = dataset.copy()  # Deep copy rather than pointer
    # Backup a test dataset without hot encoding with all error rows
    hc_testDataset_noDummy = hc_dataset.loc[(hc_dataset["household_children"] == "Don't know") | (hc_dataset["household_children"] == "Prefer not to say")]
    # Backup a training dataset without hot encoding with all error rows
    hc_trainingDataset_noDummy = hc_dataset.loc[(hc_dataset["household_children"] != "Don't know") & (hc_dataset["household_children"] != "Prefer not to say")]

    hc_hcColumnBackup = hc_dataset["household_children"]  # Backup target column
    hc_dataset.drop("household_children", axis="columns", inplace=True)  # Drop target column for one hot encoding
    hc_indexBackup = hc_dataset["Index"]  # Backup index column
    hc_dataset.drop("Index", axis="columns", inplace=True)  # Remove index column for one hot encoding
    hc_timeColumnBackup = hc_dataset["endtime"]  # Backup date time column
    hc_dataset.drop("endtime", axis="columns", inplace=True)  # Remove date time column for one hot encoding
    hc_dataset = pd.get_dummies(data=hc_dataset, drop_first=True)  # One hot encoding
    # Add back backed up columns
    hc_dataset["household_children"] = hc_hcColumnBackup
    hc_dataset["Index"] = hc_indexBackup
    hc_dataset["endtime"] = hc_timeColumnBackup

    # Create a test dataset with all error rows
    hc_testDataset = hc_dataset.loc[(hc_dataset["household_children"] == "Don't know") | (hc_dataset["household_children"] == "Prefer not to say")]
    # Create a training dataset with all error rows
    hc_trainingDataset = hc_dataset.loc[(hc_dataset["household_children"] != "Don't know") & (hc_dataset["household_children"] != "Prefer not to say")]
    hc_trainingDataset["household_children"] = hc_trainingDataset["household_children"].astype("int64")  # Convert hc to int
    # Separate features and target
    hc_x_train = hc_trainingDataset.drop(["household_children", "Index", "endtime"], axis="columns")
    hc_y_train = hc_trainingDataset["household_children"]
    hc_x_test = hc_testDataset.drop(["household_children", "Index", "endtime"], axis="columns")
    # Model Training
    lr = LinearRegression()
    lr.fit(hc_x_train, hc_y_train)
    # Predict hc_y_test
    hc_y_test = lr.predict(hc_x_test)
    max(hc_y_test)  # Make sure the max value is not greater than 5
    min(hc_y_test)  # Make sure the min value is not less than 0
    hc_y_test = hc_y_test.round(decimals=0).astype("int64")  # Round and change type to integer
    hc_testDataset_noDummy["household_children"] = hc_y_test  # Add prediction result to test data set
    # Combine no dummy test dataset and no dummy training dataset to obtain new dataset
    hc_dataset = pd.concat([hc_trainingDataset_noDummy, hc_testDataset_noDummy])
    hc_dataset.sort_values("Index", inplace=True)
    hc_dataset
    # Replace our main dataset
    dataset = hc_dataset
    dataset["household_children"] = dataset["household_children"].astype("int64")  # Change household_children to int
    dataset.dtypes

    # Change endtime column to year, month, and day
    year = []
    month = []
    day = []
    for unit in dataset["endtime"]:
        dayTemp = unit[0] + unit[1]
        yearTemp = unit[6] + unit[7] + unit[8] + unit[9]
        monthTemp = ""
        monthTempTemp = unit[3] + unit[4]
        if monthTempTemp == "01":
            monthTemp = "Jan"
        elif monthTempTemp == "02":
            monthTemp = "Feb"
        elif monthTempTemp == "03":
            monthTemp = "Mar"
        elif monthTempTemp == "04":
            monthTemp = "Apr"
        elif monthTempTemp == "05":
            monthTemp = "May"
        elif monthTempTemp == "06":
            monthTemp = "Jun"
        elif monthTempTemp == "07":
            monthTemp = "Jul"
        elif monthTempTemp == "08":
            monthTemp = "Aug"
        elif monthTempTemp == "09":
            monthTemp = "Sep"
        elif monthTempTemp == "10":
            monthTemp = "Oct"
        elif monthTempTemp == "11":
            monthTemp = "Nov"
        else:
            monthTemp = "Dec"
        year.append(yearTemp)
        month.append(monthTemp)
        day.append(dayTemp)
    # Add year, month and day to the dataset
    dataset["Year"] = year
    dataset["Month"] = month
    dataset["Day"] = day
    # Drop endtime column
    dataset.drop("endtime", axis="columns", inplace=True)
    # Rearrange columns
    cols = dataset.columns.tolist()
    cols = [cols[0]] + [cols[len(cols) - 3]] + [cols[len(cols) - 2]] + [cols[2]] + [cols[len(cols) - 1]] + [cols[1]] + cols[3:(len(cols) - 3)]
    dataset = dataset[cols]
    # Drop Index column
    dataset.drop("Index", axis="columns", inplace=True)
    # Output the cleaned dataset to csv
    dataset.to_csv("COVID_Data_Cleaned.csv", index=False)

    # Change all object type to string type
    for columnName in dataset:
        if dataset[columnName].dtype == "object":
            dataset[columnName] = dataset[columnName].astype("string")  # Change all object type to string
    a = dataset.dtypes
    # Print a table of column types
    n = 0
    for item in a:
        print(a.index[n] + " : " + str(item) + ", ", end="")
        if (n + 8) % 7 == 0:
            print("")
        n = n + 1

    # ----------------------------------- This is the end of data cleaning ---------------------------------------------

    # Model 1 training

    dataset = pd.read_csv("COVID_Data_Cleaned.csv")
    pd.options.display.max_columns = None  # Do not hide column info
    pd.options.display.max_rows = None  # Do not hide row info
    dataset.drop("Year", axis="columns", inplace=True)
    dataset.fillna("Not_Applicable", inplace=True)
    posNegData = dataset.loc[
        (dataset["i3_health"] == "Yes, and I tested negative") | (dataset["i3_health"] == "Yes, and I tested positive")]
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
    print("Model accuracy is: " + str(round(output[2], 4) * 100) + "%")
    print("Model precision is: " + str(round(output[3], 4) * 100) + "%")
    print("Model recall is: " + str(round(output[4], 4) * 100) + "%")
    print("Model false positive rate is: " + str(round(output[5], 4) * 100) + "%")
    print("Model false negative rate is: " + str(round(output[6], 4) * 100) + "%")













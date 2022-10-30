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

#------------------------------ EDA for Model 2 - Predicting Willingness to Isolate ---------------------

df = pd.read_csv('COVID_Data_Cleaned.csv')
#explore head of dataframe
df.head()

#identify how many interviewees are from Canada and US
df['RecordNo'].value_counts().plot(kind='bar')

#identify regions where there is a higher amount of interviewees
df['state_province'] = df['region_state'].str.split('/').str[0]

df['state_province'].value_counts().head(10).plot(kind='barh')

df['state_province'].value_counts().tail(10).plot(kind = 'barh')

#i9_health is the target variable - ensuring there is a decent amount of "Nos"
df['i9_health'].value_counts().plot(kind='bar')

#Age vs. willingness to isolate
sns.boxplot(x=df['i9_health'], y=df['age'])
plt.xticks(rotation = 90)
plt.xlabel("Willingness to Self-Isolate")


#Household size vs. willingness to isolate
sns.boxplot(x=df['i9_health'], y=df['household_size'])
plt.xticks(rotation = 90)
plt.xlabel("Willingness to Self-Isolate")

#Household children vs. willingness to isolate
sns.boxplot(x=df['i9_health'], y=df['household_children'])
plt.xticks(rotation = 90)
plt.xlabel("Willingness to Self-Isolate")


#house-hold_size vs. number of people in your house you have come in contact with
sns.scatterplot(x=df['household_size'], y=df['i1_health'], hue=df['i9_health'])
plt.ylabel('# of household members come in contact')
plt.legend(bbox_to_anchor=(1.05, 1))

#not expecting people to say they have come in contact with 1000 people in their house
df['i1_health'].max()

#number of people in you've come in contact with in your household should not exceed your household size?
plt.boxplot(df['i2_health'])

#looks like there are many outliers, for example coming into contact with 1000 people in your house
df.plot(kind = 'box', subplots = True, layout = (3,4), 
             sharex=False, sharey=False, fontsize=12, figsize = (18,10))

#plotting boxplots just with abnormal results
df[['i1_health','i2_health', 'i7a_health', 'i13_health', 'weight']].plot(kind = 'box', subplots = True, layout = (2,3), 
             sharex=False, sharey=False, fontsize=12, figsize = (18,10))

#percent of willingness to isolate based on cough
g = sns.countplot(df['i9_health'], hue = df['i5_health_1'])
plt.xlabel('Willingness to Isolate')
plt.legend(title = 'Cough?')
plt.show(g)
pd.crosstab(df['i5_health_1'], df['i9_health'], normalize = 'index', rownames = ['Cough?'], colnames = ['Willingness to Isolate'])

#plotting essentially a cross-tab analysis, no differences here so will do this moving forward
counts = (df.groupby(['i5_health_1'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i5_health_1', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Cough?')

#plotting willingness to isolate based on whether individual has a fever
counts = (df.groupby(['i5_health_2'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i5_health_2', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Fever?')

#plotting willingness to isolate based on whether individual has a loss of smell
counts = (df.groupby(['i5_health_3'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i5_health_3', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Loss of Smell?')


#plotting willingness to isolate based on whether individual has a loss of taste
counts = (df.groupby(['i5_health_4'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i5_health_4', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Loss of Taste?')

#plotting willingness to isolate based on whether individual has difficult breathing
counts = (df.groupby(['i5_health_5'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i5_health_5', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Difficulty Breathing?')

#plotting willingness to isolate based on an individual's ease of isolation
counts = (df.groupby(['i10_health'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i10_health', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Ease of Isolating?',bbox_to_anchor=(1.05, 1))

#plotting willingness to isolate based on whether an individual wears a mask
counts = (df.groupby(['i12_health_1'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_1', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Wear Mask?', bbox_to_anchor=(1.33, 1))

#plotting willingness to isolate based on whether an individual uses soap when washing hands
counts = (df.groupby(['i12_health_2'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_2', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Use Soap?',bbox_to_anchor=(1.33, 1))

#plotting willingness to isolate based on an individual uses hand sanitizer
counts = (df.groupby(['i12_health_3'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_3', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Use Sanitizer?',bbox_to_anchor=(1.33, 1))


#plotting willingness to isolate based on an individual covers their mouth when they cough
counts = (df.groupby(['i12_health_4'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_4', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Cover When Cough?',bbox_to_anchor=(1.33, 1))

#plotting willingness to isolate based on an individual avoids people who have been exposed to COVID
counts = (df.groupby(['i12_health_5'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_5', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Avoid Ppl Exposed to COVID?',bbox_to_anchor=(1.33, 1))

#plotting willingness to isolate based on an individual avoids going out of the house
counts = (df.groupby(['i12_health_6'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_6', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Avoid Going Out?',bbox_to_anchor=(1.33, 1))

#plotting willingness to isolate based on an individual avoids public transport
counts = (df.groupby(['i12_health_7'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_7', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Avoid Public Transport?',bbox_to_anchor=(1.33, 1))

#plotting willingness to isolate based on an individual avoids having guest over
counts = (df.groupby(['i12_health_11'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_11', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Avoid Having Guest Over?',bbox_to_anchor=(1.05, 1))

#plotting willingness to isolate based on an individual avoids small gatherings
counts = (df.groupby(['i12_health_12'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_12', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Avoid Small Gatherings?',bbox_to_anchor=(1.05, 1))

#plotting willingness to isolate based on an individual avoids large gatherings
counts = (df.groupby(['i12_health_14'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_14', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Avoid Large Gthrings?',bbox_to_anchor=(1.05, 1))

#plotting willingness to isolate based on an individual avoids crowded areas (similar to large gatherings)
counts = (df.groupby(['i12_health_15'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_15', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Avoid Crowded Areas?',bbox_to_anchor=(1.05, 1))

#plotting willingness to isolate based on an individual avoids touching public objects
counts = (df.groupby(['i12_health_20'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i12_health_20', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Avoid touching public objects?',bbox_to_anchor=(1.05, 1))

#number of times wash hands vs. willingness to isolate
sns.boxplot(x=df['i9_health'], y=df['i13_health'])
plt.xticks(rotation = 90)
plt.xlabel("Willingness to Self-Isolate")
# many outliers, no one washes 1000 times?

#plotting willingness to isolate based on an individual says no to all health issues
counts = (df.groupby(['d1_health_99'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'd1_health_99', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'No to all health issues?',bbox_to_anchor=(1.05, 1))

#plotting willingness to isolate based on an individual household size
counts = (df.groupby(['household_size'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'household_size', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Household size?',bbox_to_anchor=(1.05, 1))

#plotting willingness to isolate based on an individual's household children size
counts = (df.groupby(['household_children'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'household_children', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Household Children?',bbox_to_anchor=(1.05, 1))

#plotting willingness to isolate based on an individual's employment status
counts = (df.groupby(['employment_status'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'employment_status', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Employment Status?',bbox_to_anchor=(1.05, 1))

#plotting willingness to isolate based on an individual has any symptoms
counts = (df.groupby(['i5_health_99'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'i5_health_99', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Any symptoms?',bbox_to_anchor=(1, 1))

#plotting willingness to isolate based on an individual has any pre-existing health conditions
counts = (df.groupby(['d1_health_99'])['i9_health'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
sns.barplot(x = 'i9_health', y ='percentage', hue = 'd1_health_99', data = counts)
plt.xlabel('Willing to Isolate?')
plt.legend(title = 'Any symptoms?',bbox_to_anchor=(1, 1))


#------------------------------ Modelling for Model #2 - Predicting Willingness to Isolate ---------------------

import numpy as np
import pandas as pd
import sweetviz as sv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from geopy.geocoders import Nominatim
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import seaborn as sns

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 200)
sns.set(rc={'figure.figsize':(11.7,8.27)})

def hot_encode(df):
    for c in df.columns:
        if c=='i9_health':
            df[c].replace('Yes', 1, inplace=True)
            df[c].replace('No', 0, inplace=True)
            df[c].replace('N/A', 2, inplace=True)
            df[c].replace('Not sure', 2, inplace=True)
        else:
        
            if set(df[c].value_counts().index).issubset({'Yes', 'No', 'Not sure', 'N/A'}):
                df[c].replace('Yes', 1, inplace=True)
                df[c].replace('No', 0, inplace=True)

            if set(df[c].value_counts().index).issubset({'Always', 'Not at all', 'Sometimes', 'Frequently', 'Rarely', 'N/A'}):
                df[c].replace('Always', 2, inplace=True)
                df[c].replace('Frequently', 1, inplace=True)
                df[c].replace('Sometimes', 0, inplace=True)
                df[c].replace('Rarely', -1, inplace=True)
                df[c].replace('Not at all', -2, inplace=True)

            if set(df[c].value_counts().index).issubset({'USA', 'CAN'}):
                df[c].replace('USA', 1, inplace=True)
                df[c].replace('CAN', -1, inplace=True)

            if set(df[c].value_counts().index).issubset({'Male', 'Female'}):
                df[c].replace('Male', 1, inplace=True)
                df[c].replace('Female', -1, inplace=True)

            if set(df[c].value_counts().index).issubset({'Yes, and I tested positive', 'Yes, and I have not received my results from the test yet', 'Yes, and I tested negative', 'No, I have not', 'N/A'}):
                df[c].replace('Yes, and I tested positive', -1, inplace=True)
                df[c].replace('Yes, and I have not received my results from the test yet', 0, inplace=True)
                df[c].replace('No, I have not', 0, inplace=True)
                df[c].replace( 'Yes, and I tested negative', 1, inplace=True)

            if set(df[c].value_counts().index).issubset({'Not sure', 'Yes, and they tested positive', 'No, they have not', 'Yes, and they have not received their results from the test yet', 'Yes, and they tested negative', 'N/A'}):
                df[c].replace('Yes, and they tested positive', -1, inplace=True)
                df[c].replace('Yes, and they have not received their results from the test yet', 0, inplace=True)
                df[c].replace('No, they have not', 0, inplace=True)
                df[c].replace('Yes, and they tested negative', 1, inplace=True)

            if set(df[c].value_counts().index).issubset({'Part time employment', 'Not working', 'Unemployed', 'Other', 'Full time employment', 'Full time student', 'Retired'}):
                df[c].replace('Full time employment', 1, inplace=True)
                df[c].replace('Full time student', 1, inplace=True)
                df[c].replace('Part time employment', 0.5, inplace=True)
                df[c].replace('Not working', 0, inplace=True)
                df[c].replace('Unemployed', 0, inplace=True)
                df[c].replace('Other', 0, inplace=True)
                df[c].replace('Retired', 0, inplace=True)

            if set(df[c].value_counts().index).issubset({'Very easy', 'Somewhat difficult', 'Somewhat easy', 'Neither easy nor difficult', 'Very difficult', 'N/A', 'Not sure'}):
                df[c].replace('Very easy', 5, inplace=True)
                df[c].replace('Somewhat difficult', 2, inplace=True)
                df[c].replace('Somewhat easy', 4, inplace=True)
                df[c].replace('Neither easy nor difficult', 3, inplace=True)
                df[c].replace('Very difficult', 1, inplace=True)

            if set(df[c].value_counts().index).issubset({'Very willing', 'Somewhat willing', 'Neither willing nor unwilling', 'Somewhat unwilling', 'Very unwilling', 'N/A', 'Not sure'}):
                df[c].replace('Very willing', 5, inplace=True)
                df[c].replace('Somewhat willing', 4, inplace=True)
                df[c].replace('Neither willing nor unwilling', 3, inplace=True)
                df[c].replace('Somewhat unwilling', 2, inplace=True)
                df[c].replace('Very unwilling', 1, inplace=True)

            df[c].replace('N/A', 0, inplace=True)
            df[c].replace('Not sure', 0, inplace=True)
        
    return df

def combine_columns(df):
    temp_df = df[[c for c in df.columns if c[:3]=="i5_" and c[-3:]!="_99"]]
    df["i5_health_avg"] = temp_df.sum(axis=1)/temp_df.columns.size

    temp_df = df[[c for c in df.columns if c[:4]=="i12_"]]
    df["i12_health_avg"] = temp_df.sum(axis=1)/temp_df.columns.size

    temp_df = df[[c for c in df.columns if c[:3]=="d1_" and c[-3:]!="99"]]
    df["d1_health_avg"] = temp_df.sum(axis=1)/temp_df.columns.size

    temp_df = df[[c for c in ['i10_health', 'i11_health']]]
    df["i10_11_sum"] = temp_df.sum(axis=1)
    df["i10_11_prod"] = temp_df.prod(axis=1)
    
    return df

def get_coordinates(df):
    locs = {}
    geolocator = Nominatim(user_agent='http')
    df["region_state"] = df["region_state"].apply(lambda x: x.split(" / ")[0])

    def loc2coord(loc, geolocator):
        location = geolocator.geocode(loc)
        lat = location.latitude
        lng = location.longitude
        return lat,lng

    for loc in df["region_state"].unique():
        locs[loc] = loc2coord(loc, geolocator)
        
    df['lat'] = df["region_state"].apply(lambda x: locs[x][0])
    df['lng'] = df["region_state"].apply(lambda x: locs[x][1])
        
    return df

def tag_y(df):
    df['will_isolate'] = df['i9_health'].copy()
    df['test_data'] = df['will_isolate'].apply(lambda x: x<2)
    df['will_isolate'] = df['will_isolate'].replace(2,0)
    return df

def drop_outliers(df):
    get_z = stats.zscore(df[[c for c in df.columns if df[c].max()>2]]).apply(np.vectorize(lambda x: x if x<5 else np.nan))
    big_z = get_z[get_z.isna().any(axis=1)]
    df.drop(index=big_z.index, inplace=True)
    return df

def rescale(df):
    for c in df.iloc[:, :-1].columns:
        cmin = df[c].min()
        df[c] = df[c].apply(lambda x: x-cmin)
        if abs(df[c].max()/df[c].mean())>10:
            df[c] = df[c].apply(lambda x: np.log2(x+0.01))
    return df

def split_X_y(df, test_size=0.25):
    X = df.drop(columns=["will_isolate"])
    y = df.iloc[:,-2:]
    # print(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    X_train = X_train.iloc[:,:-1]
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    y_train = y_train["will_isolate"]

    X_test_scaled = scaler.transform(X_test[X_test["test_data"]==True].iloc[:,:-1])
    y_test = y_test[y_test["test_data"]==True]["will_isolate"]
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def optimize_precision(X_train, y_train):
    neigh = KNeighborsClassifier()
    neigh.fit(X_train, y_train)
    
    kfold = KFold(n_splits = 5, shuffle=True)
    param_grid = dict(n_neighbors=np.arange(1,11,1))
    grid = GridSearchCV(estimator=neigh, param_grid=param_grid, scoring="precision", cv=kfold)
    grid_result = grid.fit(X_train, y_train)
    means = grid_result.cv_results_['mean_test_score']
    print("Best precision: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    plt.plot(np.arange(1,11,1), means)
    plt.xlabel("Number of Neighbors K")
    plt.ylabel("Precision")

    return grid_result.best_params_['n_neighbors']

def plot_conf_matrix(X_train, y_train, X_test, y_test, best_k):
    n_neighbors=best_k
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="Spectral");  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['UNWILLING', 'ISOLATING']); ax.yaxis.set_ticklabels(['UNWILLING', 'ISOLATING']);
    
    return y_pred

def plot_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    report["ISOLATING"] = report.pop('1')
    report["UNWILLING"] = report.pop('0')
    sns.heatmap(pd.DataFrame(report).iloc[:-1, ::-1].T, annot=True, cmap="Spectral")

    return None


def process_df():
    df = pd.read_csv('COVID_Data_Cleaned.csv')
    df.fillna("N/A", inplace=True)

    # sweet_report = sv.analyze(df)
    # sweet_report.show_html('sweetviz_report.html')

    df = hot_encode(df)
    df = combine_columns(df)
    df = get_coordinates(df)
    df = tag_y(df)
    df.drop(columns=['i7b_health', 'i8_health', 'i10_health', 'i11_health',
                     'RecordNo', 'Year', 'Month', 'qweek', 
                     'Day', 'i9_health', 'region_state', 'i5_health_1', 'i5_health_2',
                     'i5_health_3', 'i5_health_4', 'i5_health_5', 'i5a_health',
                     'i5_health_99', 'i12_health_1', 'i12_health_2', 'i12_health_3',
                   'i12_health_4', 'i12_health_5', 'i12_health_6', 'i12_health_7',
                   'i12_health_8', 'i12_health_9', 'i12_health_10', 'i12_health_11',
                   'i12_health_12', 'i12_health_13', 'i12_health_14', 'i12_health_15',
                   'i12_health_16', 'i12_health_17', 'i12_health_18', 'i12_health_19',
                   'i12_health_20', 'i14_health_1', 'i14_health_2',
                   'i14_health_3', 'i14_health_4', 'i14_health_5', 'i14_health_6',
                   'i14_health_7', 'i14_health_8', 'i14_health_9', 'i14_health_10',
                   'i14_health_96', 'i14_health_98', 'i14_health_99', 'd1_health_1',
                   'd1_health_2', 'd1_health_3', 'd1_health_4', 'd1_health_5',
                   'd1_health_6', 'd1_health_7', 'd1_health_8', 'd1_health_9',
                   'd1_health_10', 'd1_health_11', 'd1_health_12', 'd1_health_13',
                   'd1_health_98', 'd1_health_99', 'weight'], inplace=True)
    df = drop_outliers(df)
    df = rescale(df)
    
    return df

df = process_df()
X_train, X_test, y_train, y_test = split_X_y(df)

best_k = optimize_precision(X_train, y_train)

y_pred = plot_conf_matrix(X_train, y_train, X_test, y_test, best_k)

plot_report(y_test, y_pred)








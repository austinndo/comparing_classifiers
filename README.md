# Comparing Classifiers

Given data that is related with direct marketing campaigns of a banking institution, we would like to compare the performance of four different classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines). The data includes information regarding customer ages, jobs, marital status, etc. and whether the client will subscribe (yes/no) a bank term deposit (indicated by variable 'y'):

1. Explore the data and verify quality
2. Make any necessary transformations via transformers and Pipelines
3. Build various models and visualize model results
4. Assess model performance

The full analysis can be found [here](prompt_III_ADo.ipynb)

## 1. Explore the data and verify quality

I decided to use the full data of 'data/bank-additional-full.csv'. The data included very little null values but it did contain some fields 'unknown'. I decided to convert those to nulls and drop all null values using `dropna()` to clean the data.

The full data contained 41188 entries while the cleaned data was reduced to 30488 entries.

![full_data](/assets/df_initial.png)

![cleaned_data](/assets/df_cleaned.png)

## 2. Make any necessary transformations via transformers and Pipelines

In order to build the models, I adjusted the values of object dtype using OneHotEncoder and a StandardScaler for the remaining data of int64. I then visualized the data using a pairplot to see how the different columns were spread by the target 'y'.

![pairplot](/assets/pairplot.png)

## 3. Build various models and visualize model results

I continued by building models for KNN, Logistic Regression, Decision Tree, and SVC. For each model I also visualized their confusion matrix and ROC curve(s).

![confusion_matrix_example](/assets/SVC_confusion_matrix.png)

![roc_curve_example](/assets/KNN_roccurve.png)

## 4. Assess model performance

While the confusion matrix and ROC curve plots are helpful to visualize the performance of the models, we can also compare their performance by looking at their accuracy scores between the train and test sets.

![train_test](/assets/train_vs_test_scores.png)

All classifiers seemed to have performed pretty well. From this dataframe we are able to compare the scores on both the train and test sets easily. SVC performs just slightly above the other classifiers with a train score of 0.909 and test score of 0.900. The SVC has the highest accuracy of the models we tested.

However, SVC did take a longer time to compute and with more or larger data we may find the other classifiers to be better suited while delivering very similar results in regards to overall accuracy.

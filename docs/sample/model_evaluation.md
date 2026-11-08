# Evaluating machine learning models

To judge whether a model will work on new data, split the dataset into a training set and a test set. The model learns from
the training data and is scored only on the held-out test data. Scoring on the training data alone is misleading because of
overfitting: an overfit model memorises noise in the training set and performs poorly on unseen examples.

For regression problems, common metrics are root mean squared error (RMSE), mean absolute error and R squared, which is the
fraction of variance explained by the model. For classification, accuracy can be misleading when classes are imbalanced, so
precision, recall and the F1 score are also used. Precision is the share of predicted positives that are correct, and recall is
the share of actual positives that were found.

Cross-validation repeats the train and test split several times and averages the scores, giving a more stable estimate than a
single split. Regularisation, simpler models and more data all help to reduce overfitting. Always compare a model against a simple
baseline, such as predicting the mean, to make sure the added complexity is worthwhile.

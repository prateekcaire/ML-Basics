# ML-Basics
### Python implementation of basic ML algorithms from scratch using numpy and pandas

## 1. K Nearest Neighbors (KNN)
- KNN is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions).
- Iris Dataset: ![KNN Dataset](images/knn.png)

## 2. Linear Regression
- Linear regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables).
- Chart: 
![Linear Regression](images/linreg.png)
- Equations: 
![Linear regression equations](images/linearreg_equations.png)


## 3. Logistic Regression
- Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable.
- Model: 
![Logistic Regression](images/logreg.png)
- Equations: 
![Linear regression equations 1](images/logreg_eq_1.png) ![Linear regression equations 2](images/logreg_eq_2.png)
## 4. Decision Trees
- Implementation is done to create a binary for a specific value among all continuous feature value
- Most optimal feature, split pair is found where entropy is minimum 
- Entropy H(p) = -plog(p) -(1-p)log(p). Same as cross entropy loss
- when best feature, split pair is found, tree node with subset of X, subset of y, and feature index is created
- above process is repeated unless entropy is 0 or max depth is reached
- label from leaf node is used as prediction. If leaf node is pure, single label is returned, else most common label is predicted
- Chart: ![Decision Trees 1](images/DecisionTree/chart19.png), ![Decision Trees 2](images/DecisionTree/chart23.png), ![Decision Trees 3](images/DecisionTree/chart29.png)

## 5. Random Forest
- Same as Decision Tree but multiple decision trees are created using random subset of features
- prediction is made by majority voting for classification or mean for regression
- since subset of features are used, correlation is reduced among features. Also variance is reduced. hence less overfitting 
- Chart: ![Random Forest](images/rand_forest.png)

## 6. Naive Bayes
- Predict the posterior, P(y|X), by naively assuming that all features have zero correlation
- Each independent class conditional probability, P(x_i|y) is modeled as gaussian and its parameters (mean and variance) are pre-calculated for each classification
- Prediction finds the argmax_y of sum of prior(log(P(y))) and likelihoods sum. Quite basic
- Posterior Probability Equation: ![Naive Bayes](images/naive-bayes.png)
- Chart: ![Naive Bayes Chart](images/naive_bayes_visualization.png)

## 7. Support Vector Machine
- Finds the optimal hyperplane that best separates the different classes in feature space
- hyperplane is defined as: w·x + b = 0
- goal is to maximize the margin while ensuring all points are classified correctly
- optimization problem that minimize ||w|| subject to y_i(w·x_i + b) ≥ 1 for all i
- Chart: ![SVM](images/svm.png)

## 8. K-Means
- iteratively label the data point and find centroid until centroid converges 
- Chart: ![kmeans1](images/k-means/fig0.png)![kmeans2](images/k-means/fig3.png)![kmeans1](images/k-means/fig6.png)![kmeans9](images/k-means/fig9.png)

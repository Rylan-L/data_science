# Data Science Examples and Notebooks by Rylan Larsen
---
This is a compilation of notebooks which I created for data analysis or for exploration of machine learning algorithms. This includes everything from data visualization and wrangling, feature selection, to classifiers, and neural networks.
<br/>
<br/>
## In-depth examples
---
## Can you predict the weather?
[Github](https://github.com/Rylan-L/data_science/blob/master/machine_learning/neural_networks/7202019_Rain_in_NY_Model.ipynb) [NBViewer](https://nbviewer.jupyter.org/github/Rylan-L/data_science/blob/master/machine_learning/neural_networks/7202019_Rain_in_NY_Model.ipynb)

Given historical weather data, can I predict whether tomorrow will be rainy in New York? In this example, I use historical weather data from NY to ask, given previous days, can I predict whether any next day (tomorrow) be rainy? In this example, I largely use LSTM neural networks and Keras/Tensorflow to address this.


Datasource: Kaggle, SelfishGene, https://www.kaggle.com/selfishgene/historical-hourly-weather-data
<br/>
<br/>
## Semiconductor manufacturing sensors: Data cleaning, feature selection, and machine learning on 

One useful data science approach in manufacturing is to make predictions about the pass/fail or quality of a product given sensor readouts from the manufacturing line. The SECOM dataset has a large number of such sensor readings and pass/fail labels for Semiconductor manufacturing. However, the data set suffers from some common problems

* Feature selection: the data has large feature space (591), of which many sensor readings (features) are not actually useful in predicting the pass/fail of a product
* Class imbalance: the data has a large number of positive (pass) examples, but few fail examples to train a classifier on. Therefore most classifiers do much better on the positive (pass) cases than on the failures. Arguably the failure  cases are more important to the Semiconductor company, and a model must be sensitive to these.
* Small Dataset: there are only 1567 examples in this dataset, giving us few examples to optimize our machine learning training sets on.

### Part 1: Data cleaning, dealing with class imbalance and feature selection. Manufacturing pass/fail classification with SVM, decision trees, and random forests.
[Github](https://github.com/Rylan-L/data_science/blob/master/machine_learning/neural_networks/DataSci_example_Secom_pt1.ipynb) [NBViewer](https://nbviewer.jupyter.org/github/Rylan-L/data_science/blob/master/machine_learning/neural_networks/DataSci_example_Secom_pt1.ipynb)

In the first notebook, I deal with missing data, perform feature selection using three methods (a wrapper, filter, and Lasso/L1 embedded approach), and deal with class imbalance using Synthetic Minority Over-sampling Technique (SMOTE). Then SVM, decision trees, and random forests classifiers were used for manufacturing classification

### Part 2: Using Neural Networks for classification of failing products 
[Github](https://github.com/Rylan-L/data_science/blob/master/machine_learning/neural_networks/DataSci_example_Secom_pt2.ipynb) [NBViewer](https://nbviewer.jupyter.org/github/Rylan-L/data_science/blob/master/machine_learning/neural_networks/DataSci_example_Secom_pt2.ipynb)

As a follow-up to using non-neural network classifiers, I tested three neural network models classifiers (simple neural network, deep neural network, LSTM neural network) on the Secom dataset. The goal here was to make a classifier which predicted the class (pass/fail) for the manufacturing product. 
<br/>
<br/>
## Classification with Neural Networks
---
### Text Analytics using LSTM neural networks
[Github](https://github.com/Rylan-L/data_science/blob/master/machine_learning/neural_networks/MultiClass_Text_prediction_LSTM_NN.ipynb) [NBViewer](https://nbviewer.jupyter.org/github/Rylan-L/data_science/blob/master/machine_learning/neural_networks/MultiClass_Text_prediction_LSTM_NN.ipynb)

Neural networks can be used for text prediction and natural language processing. Keras includes a dataset of 11,228 newswires from Reuters, with labeled over 46 topics. Predicting the topic depends not just on the the previous words, but the sequence in which they are presented. Therefore they are an ideal test case for LSTM neural networks which take sequences as inputs. 
<br/>
<br/>
## Classification with other machine learning approaches 
---
### Predicting breast cancer using decision trees

[Colloboratory](https://colab.research.google.com/drive/1hL9ZE3pvvJgmg3eXi1x97xij4SIam9eh) [Github](https://github.com/Rylan-L/data_science/blob/master/machine_learning/decision_trees_cancer.ipynb) [NBViewer](https://nbviewer.jupyter.org/github/Rylan-L/data_science/blob/master/machine_learning/decision_trees_cancer.ipynb)

One of the potential transformative applications of machine learning is in the use of medical diagnostics. In this notebook, I explored using decision trees using Gini Indexing or Entropy for splitting. This notebook also demonstrates decision tree visualization, allowing the user to see which attributes and values are useful for predicting cancer in this dataset.

### Predicting a bank customer's purchase using demographics: Random Forests, Gradient-Boosted Decision Trees

[Colloboratory](https://colab.research.google.com/drive/1IlDlP8giHJnWLwf10Z1U1rzQX6ZSbQTn) [Github](https://github.com/Rylan-L/data_science/blob/master/machine_learning/random_forests_gradient_boosted_dtrees_bank_customers.ipynb) [NBViewer](https://nbviewer.jupyter.org/github/Rylan-L/data_science/blob/master/machine_learning/random_forests_gradient_boosted_dtrees_bank_customers.ipynb) 

A useful business application of machine learning is to predict a customers purchase choice based on their demographics. In this notebook, I explore random forests and gradient-boosted decision trees for predicting whether bank customers will purchase a specific product.

### Support Vector Machine prediction of the age of Abalone Shellfish 
[Github](https://github.com/Rylan-L/data_science/blob/master/machine_learning/SVM_abalone.ipynb) [NBViewer](https://nbviewer.jupyter.org/github/Rylan-L/data_science/blob/master/machine_learning/SVM_abalone.ipynb) 

I demonstrate the use support vector machines (SVM) for predicting the number of rings in abalaone shells using other parameters in the dataset such as length and weight. The number of rings are related to the age of the abalone. This notebook also demonstrates the use of GridsearchCV for hyperparameter tuning.

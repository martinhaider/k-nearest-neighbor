# K-Nearest Neighbors Algorithm
This example shows a simple implementation of the k-nearest neighbors algorithm. First the data is plotted in a scatter plot. Next the model is trained and the generalization error is calculated over the different ks. To test the model the 10-fold cross validation is used.
In a next step some noise is added to the data to compare the difference of the prediction error rate with and without noise. The probability of label change is 1 out of 5.
To test the algorithm in higher dimensions random columns are added to the data set. First one, then two, three and four columns are added and the generalization error is calculated.
Of course in every step the solution is plotted.

## Run algorithm
```
python k-nearest-neighbor
```

## Versions
```
python 3.7
pip 19.1.1
```

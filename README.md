# H2O-AutoML-Evaluation

### Question

Will H2O AutoML give high scores on Kaggle datasets?

### Hypothesis

With minimal effort, the H2O AutoML will be in the top 10% of the leaderboard for Titanic, the MNIST dataset, and the housing prices dataset.

### Data Sources

- https://www.kaggle.com/c/titanic
- https://www.kaggle.com/c/digit-recognizer
- https://www.kaggle.com/c/house-prices-advanced-regression-techniques

### Data Descriptions

##### Descriptions from https://www.kaggle.com/c/titanic
- survival:	Survival	0 = No, 1 = Yes
- pclass:	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
- sex:	Sex
- Age:	Age in years
- sibsp:	# of siblings / spouses aboard the Titanic
- parch:	# of parents / children aboard the Titanic
- ticket:	Ticket number
- fare:	Passenger fare
- cabin:	Cabin number
- embarked:	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

##### Descriptions from https://www.kaggle.com/c/digit-recognizer
- Pixel000-780: pixel color on grayscale (0-255)

##### Descriptions from https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- There are a lot of fields

### Steps

- Titanic
  - Improve Algorithms
    - Follow Auto H2O Documentation
    - Train for 5 minutes

- MNST
  - Improve Algorithms
    - Follow Auto H2O Documentation
    - Train for 5 minutes  

- Housing Prices
  - Improve Algorithms
    - Follow Auto H2O Documentation
    - Train for 5 minutes

### Results

- Target: Top 10% of submissions
- Titanic: 76% accuracy (In the top 79% of submissions)
- MNIST: 96.1% accuracy (In the top 80% of submissions)
- Housing: 0.13 score (In the top 44% of submissions)

### Conclusion

Simply using the H2O AutoML with 5 minutes of training isn't enough to get into the top 10% of submissions.

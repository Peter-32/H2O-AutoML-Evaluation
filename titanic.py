import pandas as pd
import numpy as np
from pandasql import sqldf
import h2o
from h2o.automl import H2OAutoML


def q(q): return sqldf(q, globals())


h2o.init()

train = h2o.import_file("/Users/peterjmyers/Work/H2O-AutoML-Evaluation/data/titanic/train.csv")
test = h2o.import_file("/Users/peterjmyers/Work/H2O-AutoML-Evaluation/data/titanic/test.csv")
train, test


# Identify predictors and response
x = train.columns
y = "Survived"
x.remove(y)
x

# For binary classification, response should be a factor
train[y] = train[y].asfactor()

# Run AutoML for 10 minutes
aml = H2OAutoML(max_runtime_secs = 600)
aml.train(x = x, y = y,
          training_frame = train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb

# The leader model is stored here
aml.leader

# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly

preds = aml.predict(test)

# or:
preds = aml.leader.predict(test)

preds

type(preds)

test.head()

predictions = test.as_data_frame()[['PassengerId']]
predictions['Survived'] = preds.as_data_frame()[['predict']]
predictions

predictions.to_csv("/Users/peterjmyers/Work/H2O-AutoML-Evaluation/data/titanic/predictions.csv", index=False)

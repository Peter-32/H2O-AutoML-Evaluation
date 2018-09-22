import pandas as pd
import numpy as np
from pandasql import sqldf
import h2o
from h2o.automl import H2OAutoML


def q(q): return sqldf(q, globals())


h2o.init()

train = h2o.import_file("/Users/peterjmyers/Work/H2O-AutoML-Evaluation/data/MNIST/train.csv")
test = h2o.import_file("/Users/peterjmyers/Work/H2O-AutoML-Evaluation/data/MNIST/test.csv")
train, test


# Identify predictors and response
x = train.columns
y = "label"
x.remove(y)
x

# For binary classification, response should be a factor
train[y] = train[y].asfactor()

# Run AutoML for 5 minutes
aml = H2OAutoML(max_runtime_secs = 300)
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

predictions = pd.DataFrame()
predictions['Label'] = preds.as_data_frame()[['predict']]
predictions['ImageId'] = predictions.index
predictions['ImageId'] = predictions['ImageId'].apply(lambda x: x + 1)
predictions = q("select ImageId, label from predictions")
predictions

predictions.to_csv("/Users/peterjmyers/Work/H2O-AutoML-Evaluation/data/MNIST/predictions.csv", index=False)


# Notify Peter
import requests
msg = "done"
requests.post("https://hooks.slack.com/services/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",json={ "text": "{}: {}".format("DATABRICKS", msg)})


"""DNN Regressor with custom input_fn for Housing dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf
#将日志记录详细信息设置为INFO以获取更详细的日志输出
tf.logging.set_verbosity(tf.logging.INFO)
#Define the column names for the data set in COLUMNS
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
#To distinguish features from the label, also define FEATURES and LABEL
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

#To pass input data into the regressor, write a 工厂方法 that accepts a pandas Dataframe and returns an input_fn:
def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)


def main(unused_argv):
  # Load datasets into pandas DataFrame
  training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
  test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)

  # Set of 6 examples for which to predict median house values
  prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

  # create a list of FeatureColumns for the input data, which formally specify the set of features to use for training
  feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                        hidden_units=[10, 10],
                                        model_dir="/tmp/boston_model")

  # Train：the input_fn keeps returning data until the required number of train steps is reached
  # shuffle = True
  regressor.train(input_fn=get_input_fn(training_set), steps=5000)

  # Evaluate loss over one epoch of test_set：
  # the input_fn will iterate over the data once and then raise OutOfRangeError
  ev = regressor.evaluate(
      input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # Print out predictions over a slice of prediction_set.
  y = regressor.predict(
      input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
  # .predict() returns an iterator of dicts; convert to a list and print
  # predictions
  predictions = list(p["predictions"] for p in itertools.islice(y, 6))
  print("Predictions: {}".format(str(predictions)))

if __name__ == "__main__":
  tf.app.run()
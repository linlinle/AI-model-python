# -*- coding: utf-8 -*-

import tensorflow as tf

#   FeatureColumn handles the conversion of categorical values into vectors automatically,
eye_color = tf.feature_column.categorical_column_with_vocabulary_list(
    "eye_color", vocabulary_list=["blue", "brown", "green"])
education = tf.feature_column.categorical_column_with_hash_bucket(
    "education", hash_bucket_size=1000)


#   Feature Crosses
# creating a new feature
sport_x_city = tf.feature_column.crossed_column(
    ["sport", "city"], hash_bucket_size=int(1e4))



#   Continuous columns
age = tf.feature_column.numeric_column("age")
#   Bucketization turns a continuous column into a categorical column.
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])



#   Linear estimators
e = tf.estimator.LinearClassifier(
    feature_columns=[
        native_country, education, occupation, workclass, marital_status,
        race, age_buckets, education_x_occupation,
        age_buckets_x_race_x_occupation],
    model_dir=YOUR_MODEL_DIRECTORY)
e.train(input_fn=input_fn_train, steps=200)
# Evaluate for one step (one pass through the test data).
results = e.evaluate(input_fn=input_fn_test)
# Print the stats for the evaluation.
for key in sorted(results):
    print("%s: %s" % (key, results[key]))



#   Wide and deep learning
#This novel approach combines the ability of linear models to "memorize" key features with the generalization ability of neural nets.
e = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=YOUR_MODEL_DIR,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
#Getting Started With Tensorflow


####_Mnist_：    在 TensorFlow上训练MNIST的不同方法
* MNIST_Experts.py
[basic implementation](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_softmax.py)
and
[improve the accuracy](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_deep.py):

    * Create a softmax regression function that is a model for recognizing MNIST digits, based on looking at every pixel in the image
    * Use Tensorflow to train the model to recognize digits by having it "look" at thousands of examples (and run our first Tensorflow session to do so)
    * Check the model's accuracy with our test data
    * Build, train, and test a multilayer convolutional neural network to improve the results
    
    
* fully_connected_feed.py[code](https://www.github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/fully_connected_feed.py)

    The goal of this tutorial is to show how to use TensorFlow to train and evaluate a simple feed-forward neural network
     for handwritten digit classification using the (classic) MNIST data set.

####_TensorBoard_：  TensorBoard的实现

* Basic_Example_Histogram .py
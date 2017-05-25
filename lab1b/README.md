Both libraries we will be using are for Python. If you are using your own machine/laptop to do this work I strongly recommend you install the [Anaconda](https://www.continuum.io/downloads) distribution for Python, since this automatically includes a lot of libraries that we require, such as Numpy.

Theano
----

For those that want to learn Theano, there is a nice tutorial landing page here:

http://deeplearning.net/tutorial/

Before looking at other frameworks that wrap Theano, I strongly suggest you run through these three examples:

* [Logistic regression](http://deeplearning.net/tutorial/logreg.html#logreg)
* [Multilayer perceptron](http://deeplearning.net/tutorial/mlp.html#mlp)
* [Convolutional network](http://deeplearning.net/tutorial/lenet.html#lenet)

The logistic regression tutorial is similar to what I presented in the last lab.

If you are using your own personal machine/laptop please install the latest version of Theano via the Github page (i.e. *not* through pip's own package repository):

```
git clone https://github.com/Theano/Theano.git
cd Theano
python setup.py install
```

TensorFlow
---

For the folks interested in TensorFlow:

* [Logistic regression](https://www.tensorflow.org/get_started/mnist/beginners)
* [Convolutional network](https://www.tensorflow.org/get_started/mnist/beginners)

There are also some excellent IPython notebook tutorials too, which you can find here:

https://github.com/aymericdamien/TensorFlow-Examples/tree/master/notebooks

* [Linear regression](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb)
* [Logistic regression](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/logistic_regression.ipynb)
* [Multilayer perceptron](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/multilayer_perceptron.ipynb)

For those wanting to use TensorFlow on their own machines/laptops, installation instructions:

https://www.tensorflow.org/install/

Other information
---

IPython notebooks are a great way to experiment with libraries and allow you to easily annotate and document your code. If it is installed in the lab machines, `cd` into a directory and run `ipython notebook`.

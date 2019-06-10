---
Title : Beyond Hello World, A Computer Vision Example
category : [TensorFlow, machine Learning]
Tag : [image recognition]
---



# Beyond Hello World, A Computer Vision Example

In the previous exercise you saw how to create a neural network that figured out the problem you were trying to solve. This gave an explicit example of learned behavior. Of course, in that instance, it was a bit of overkill because it would have been easier to write the function Y=2x-1 directly, instead of bothering with using Machine Learning to learn the relationship between X and Y for a fixed set of values, and extending that for all values.

But what about a scenario where writing rules like that is much more difficult -- for example a computer vision problem? Let's take a look at a scenario where we can recognize different items of clothing, trained from a dataset containing 10 different types.



## Start Coding

Let's start with our import of TensorFlow

In [1]:

```python
import tensorflow as tf
print(tf.__version__)
```



```
1.13.0-rc1
```



The Fashion MNIST data is available directly in the tf.keras datasets API. You load it like this:

In [0]:

```
mnist = tf.keras.datasets.fashion_mnist
```



Calling load_data on this object will give you two sets of two lists, these will be the training and testing values for the graphics that contain the clothing items and their labels.

In [3]:

```
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
```



```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
32768/29515 [=================================] - 0s 0us/step
40960/29515 [=========================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26427392/26421880 [==============================] - 0s 0us/step
26435584/26421880 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
16384/5148 [===============================================================================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4423680/4422102 [==============================] - 0s 0us/step
4431872/4422102 [==============================] - 0s 0us/step
```



What does these values look like? Let's print a training image, and a training label to see...Experiment with different indices in the array. For example, also take a look at index 42...that's a a different boot than the one at index 0

In [5]:

```
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
```



```
9
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   1   0   0  13  73   0
    0   1   4   0   0   0   0   1   1   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   3   0  36 136 127  62
   54   0   0   0   1   3   4   0   0   3]
 [  0   0   0   0   0   0   0   0   0   0   0   0   6   0 102 204 176 134
  144 123  23   0   0   0   0  12  10   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0 155 236 207 178
  107 156 161 109  64  23  77 130  72  15]
 [  0   0   0   0   0   0   0   0   0   0   0   1   0  69 207 223 218 216
  216 163 127 121 122 146 141  88 172  66]
 [  0   0   0   0   0   0   0   0   0   1   1   1   0 200 232 232 233 229
  223 223 215 213 164 127 123 196 229   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0 183 225 216 223 228
  235 227 224 222 224 221 223 245 173   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0 193 228 218 213 198
  180 212 210 211 213 223 220 243 202   0]
 [  0   0   0   0   0   0   0   0   0   1   3   0  12 219 220 212 218 192
  169 227 208 218 224 212 226 197 209  52]
 [  0   0   0   0   0   0   0   0   0   0   6   0  99 244 222 220 218 203
  198 221 215 213 222 220 245 119 167  56]
 [  0   0   0   0   0   0   0   0   0   4   0   0  55 236 228 230 228 240
  232 213 218 223 234 217 217 209  92   0]
 [  0   0   1   4   6   7   2   0   0   0   0   0 237 226 217 223 222 219
  222 221 216 223 229 215 218 255  77   0]
 [  0   3   0   0   0   0   0   0   0  62 145 204 228 207 213 221 218 208
  211 218 224 223 219 215 224 244 159   0]
 [  0   0   0   0  18  44  82 107 189 228 220 222 217 226 200 205 211 230
  224 234 176 188 250 248 233 238 215   0]
 [  0  57 187 208 224 221 224 208 204 214 208 209 200 159 245 193 206 223
  255 255 221 234 221 211 220 232 246   0]
 [  3 202 228 224 221 211 211 214 205 205 205 220 240  80 150 255 229 221
  188 154 191 210 204 209 222 228 225   0]
 [ 98 233 198 210 222 229 229 234 249 220 194 215 217 241  65  73 106 117
  168 219 221 215 217 223 223 224 229  29]
 [ 75 204 212 204 193 205 211 225 216 185 197 206 198 213 240 195 227 245
  239 223 218 212 209 222 220 221 230  67]
 [ 48 203 183 194 213 197 185 190 194 192 202 214 219 221 220 236 225 216
  199 206 186 181 177 172 181 205 206 115]
 [  0 122 219 193 179 171 183 196 204 210 213 207 211 210 200 196 194 191
  195 191 198 192 176 156 167 177 210  92]
 [  0   0  74 189 212 191 175 172 175 181 185 188 189 188 193 198 204 209
  210 210 211 188 188 194 192 216 170   0]
 [  2   0   0   0  66 200 222 237 239 242 246 243 244 221 220 193 191 179
  182 182 181 176 166 168  99  58   0   0]
 [  0   0   0   0   0   0   0  40  61  44  72  41  35   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]]
```



![img](data:image/png;base64,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%0A9NKtQ3PMtoM5ewvuHSP2MGRvxh7yiot72KlrxJ7S+9Aue5notkW/NOs//3y8CZ//L1bnHko+mJ1s%0Atr1hsr00N2A/Z3d8a72zdkp1l9n2hYHZZv3zkCm/TYlesz430e2s/SD5kdl2DdxDfceCR34iTzH8%0ARJ5i+Ik8xfATeYrhJ/IUw0/kKYafyFPlHecXe3nuxf+80Wy+JPmhszao9hTKsHH8sHFby5Qqe5nm%0A4bT9MHel7Sm7YU6v2e+sXd+w2Wy7/rHFZv3S1I/N+idX/KdZbxtyb2XdnbF/7pt2XWHWN33WbNYv%0AnLvLWTsnuddsG3ZuRTKeMuvWNGsAGMi5f1/fSdnnP5QKj/xEnmL4iTzF8BN5iuEn8hTDT+Qphp/I%0AUww/kafKunR33axmPfXWv3fWW+/8V7P9Mz0XOmvNtfZeoidXHzDr0+P2ds+WZMwe8z0jYY/5vjBw%0Akll/7dCZZv07yQ5nLSH29t6XT/rYrN/207vNeqbWXiW6b677+JKpt3/3Gs49aNZ/fNqrZr3a+NkP%0AZe1x/LDHLWwL7jDWGgzJmL0t+kPLrnfW3u54Cr1D+7h0NxG5MfxEnmL4iTzF8BN5iuEn8hTDT+Qp%0Ahp/IU6Hz+UWkGcDTAJoAKIBWVX1URBoB/AbAXAAdAG5UVXPP5FgamNTpHt98oW+h2ZdT6txrnR9I%0A2+vT/+HwOWb9pDp7u2drq+nTjPn0ALA5NdWsv9j9bbN+Yp29fn1neoqzdjBdb7YdNOaVA8CTjzxs%0A1h/qtNf9v75xk7N2brU9jn8oZx+btobsd9Cfq3XWUmqv79Abch5A0vh9AIC02tGKG1t8T43Z5xD0%0AnTPdWct25r9ERz5H/gyAu1V1AYALAdwpIgsA3AugTVXnA2gLviai40Ro+FV1n6puCi73A9gGYA6A%0A5QBWBzdbDeC6ieokEZXeMb3nF5G5AM4DsAFAk6ruC0r7Mfq2gIiOE3mHX0QmA/gdgJ+o6hFvQnV0%0AgsC4J2qLyEoRaReR9szwQFGdJaLSySv8IpLAaPB/parPBVd3isjsoD4bwLg7H6pqq6q2qGpLVY39%0A4RMRlU9o+EVEADwJYJuqjv3ody2AFcHlFQCeL333iGii5DMucAmAWwFsEZEv14G+D8ADAP5bRG4H%0A8CmAG8O+UXwkh+TuYWc9p/ZMxFcPuKe2NtX2m20XJneb9e2D9rDRlqETnbVNVd8y29bF3dt7A8CU%0A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%0AYII=%0A)



You'll notice that all of the values in the number are between 0 and 255. If we are training a neural network, for various reasons it's easier if we treat all values as between 0 and 1, a process called '**normalizing**'...and fortunately in Python it's easy to normalize a list like this without looping. You do it like this:

In [0]:

```
training_images  = training_images / 255.0
test_images = test_images / 255.0
```



Now you might be wondering why there are 2 sets...training and testing -- remember we spoke about this in the intro? The idea is to have 1 set of data for training, and then another set of data...that the model hasn't yet seen...to see how good it would be at classifying values. After all, when you're done, you're going to want to try it out with data that it hadn't previously seen!



Let's now design the model. There's quite a few new concepts here, but don't worry, you'll get the hang of them.

In [0]:

```python
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
```



**Sequential**: That defines a SEQUENCE of layers in the neural network

**Flatten**: Remember earlier where our images were a square, when you printed them out? Flatten just takes that square and turns it into a 1 dimensional set.

**Dense**: Adds a layer of neurons

Each layer of neurons need an **activation function** to tell them what to do. There's lots of options, but just use these for now.

**Relu** effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.

**Softmax** takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!



The next thing to do, now the model is defined, is to actually build it. You do this by compiling it with an optimizer and loss function as before -- and then you train it by calling **model.fit** asking it to fit your training data to your training labels -- i.e. have it figure out the relationship between the training data and its actual labels, so in future if you have data that looks like the training data, then it can make a prediction for what that data would look like.

In [9]:

```
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
```



```
Epoch 1/5
60000/60000 [==============================] - 6s 103us/sample - loss: 0.2832 - acc: 0.8944
Epoch 2/5
60000/60000 [==============================] - 6s 100us/sample - loss: 0.2700 - acc: 0.8993
Epoch 3/5
60000/60000 [==============================] - 6s 96us/sample - loss: 0.2586 - acc: 0.9038
Epoch 4/5
60000/60000 [==============================] - 5s 83us/sample - loss: 0.2497 - acc: 0.9076
Epoch 5/5
60000/60000 [==============================] - 6s 99us/sample - loss: 0.2401 - acc: 0.9098
```

Out[9]:

```
<tensorflow.python.keras.callbacks.History at 0x7fe742abae50>
```



Once it's done training -- you should see an accuracy value at the end of the final epoch. It might look something like 0.9098. This tells you that your neural network is about 91% accurate in classifying the training data. I.E., it figured out a pattern match between the image and the labels that worked 91% of the time. Not great, but not bad considering it was only trained for 5 epochs and done quite quickly.

But how would it work with unseen data? That's why we have the test images. We can call model.evaluate, and pass in the two sets, and it will report back the loss for each. Let's give it a try:

In [10]:

```
model.evaluate(test_images, test_labels)
```



```
10000/10000 [==============================] - 1s 57us/sample - loss: 0.3306 - acc: 0.8838
```

Out[10]:

```
[0.33057239087224005, 0.8838]
```



For me, that returned a accuracy of about .8838, which means it was about 88% accurate. As expected it probably would not do as well with *unseen* data as it did with data it was trained on! As you go through this course, you'll look at ways to improve this.

To explore further, try the below exercises:



# Exploration Exercises



### Exercise 1:

For this first exercise run the below code: It creates a set of classifications for each of the test images, and then prints the first entry in the classifications. The output, after you run it is a list of numbers. Why do you think this is, and what do those numbers represent?

In [0]:

```python
classifications = model.predict(test_images)

print(classifications[0])
```



Hint: try running print(test_labels[0]) -- and you'll get a 9. Does that help you understand why this list looks the way it does?

In [0]:

```python
print(test_labels[0])
```



### What does this list represent?

1. It's 10 random meaningless values
2. It's the first 10 classifications that the computer made
3. It's the probability that this item is each of the 10 classes



#### Answer:

The correct answer is (3)

The output of the model is a list of 10 numbers. These numbers are a probability that the value being classified is the corresponding value, i.e. the first value in the list is the probability that the handwriting is of a '0', the next is a '1' etc. Notice that they are all VERY LOW probabilities.

For the 7, the probability was .999+, i.e. the neural network is telling us that it's almost certainly a 7.



### How do you know that this list tells you that the item is an ankle boot?

1. There's not enough information to answer that question
2. The 10th element on the list is the biggest, and the ankle boot is labelled 9
3. The ankle boot is label 9, and there are 0->9 elements in the list



#### Answer

The correct answer is (2). Both the list and the labels are 0 based, so the ankle boot having label 9 means that it is the 10th of the 10 classes. The list having the 10th element being the highest value means that the Neural Network has predicted that the item it is classifying is most likely an ankle boot



## Exercise 2:

Let's now look at the layers in your model. Experiment with different values for the dense layer with 512 neurons. What different results do you get for loss, training time etc? Why do you think that's the case?

In [0]:

```python
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
```



### Question 1. Increase to 1024 Neurons -- What's the impact?

1. Training takes longer, but is more accurate
2. Training takes longer, but no impact on accuracy
3. Training takes the same time, but is more accurate



#### Answer

The correct answer is (1) by adding more Neurons we have to do more calculations, slowing down the process, but in this case they have a good impact -- we do get more accurate. That doesn't mean it's always a case of 'more is better', you can hit the law of diminishing returns very quickly!



## Exercise 3:

What would happen if you remove the Flatten() layer. Why do you think that's the case?

You get an error about the shape of the data. It may seem vague right now, but it reinforces the rule of thumb that the first layer in your network should be the same shape as your data. Right now our data is 28x28 images, and 28 layers of 28 neurons would be infeasible, so it makes more sense to 'flatten' that 28,28 into a 784x1. Instead of wriitng all the code to handle that ourselves, we add the Flatten() layer at the begining, and when the arrays are loaded into the model later, they'll automatically be flattened for us.

In [0]:

```python
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([#tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
```



## Exercise 4:

Consider the final (output) layers. Why are there 10 of them? What would happen if you had a different amount than 10? For example, try training the network with 5

You get an error as soon as it finds an unexpected value. Another rule of thumb -- the number of neurons in the last layer should match the number of classes you are classifying for. In this case it's the digits 0-9, so there are 10 of them, hence you should have 10 neurons in your final layer.

In [0]:

```python
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
```



## Exercise 5:

Consider the effects of additional layers in the network. What will happen if you add another layer between the one with 512 and the final layer with 10.

Ans: There isn't a significant impact -- because this is relatively simple data. For far more complex data (including color images to be classified as flowers that you'll see in the next lesson), extra layers are often necessary.

In [0]:

```python
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
```

In [0]:

```

```



# Exercise 6:

Consider the impact of training for more or less epochs. Why do you think that would be the case?

Try 15 epochs -- you'll probably get a model with a much better loss than the one with 5 Try 30 epochs -- you might see the loss value stops decreasing, and sometimes increases. This is a side effect of something called 'overfitting' which you can learn about [somewhere] and it's something you need to keep an eye out for when training neural networks. There's no point in wasting your time training if you aren't improving your loss, right! :)

In [0]:

```python
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=30)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[34])
print(test_labels[34])
```



# Exercise 7:

Before you trained, you normalized the data, going from values that were 0-255 to values that were 0-1. What would be the impact of removing that? Here's the complete code to give it a try. Why do you think you get different results?

In [0]:

```python
import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
```



# Exercise 8:

Earlier when you trained for extra epochs you had an issue where your loss might change. It might have taken a bit of time for you to wait for the training to do that, and you might have thought 'wouldn't it be nice if I could stop the training when I reach a desired value?' -- i.e. 95% accuracy might be enough for you, and if you reach that after 3 epochs, why sit around waiting for it to finish a lot more epochs....So how would you fix that? Like any other program...you have callbacks! Let's see them in action...

In [0]:

```python
import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
```
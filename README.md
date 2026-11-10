Detecting the presence of a cat in images, taken from [kaggle](kaggle.com/datasets/muhammeddalkran/catvnoncat), using a variety of neural network constructions.

The data comes as two sets, train and test. The test set is further stratified split (sklearn.model_selection.train_test_split) into two, the test and validate set. The true label count across the three sets are:
- Train: 72 / 209
- Validate: 16 / 25
- Test: 17 / 25

Different neural network models are constructed (torch.nn.Module) starting from the most simplistic (net_0) being a fully connected network with no hidden layer, to those that utilizes convolutional layers (net_2), and those that are described in literature (net_5) (Springenberg 2014). A more complete description of each can be found in the starting section, in each within "model".

All constructions and its various hyperparameters are trained on the train set and validated on the validate set. Binary cross entropy loss and stochastic gradient descent are used, along with the same number weight randomization. The learn rates are searched in factors of 10. The performance of each construction and its hyperparameters, on the train and test sets, per epoch, are in "result". All are plotted and are in "plot".

For the train and validate set given, the baseline cross entropy (memorization of the train set probability without distinction of the validate set) is 
```math
-\left[
\frac{16}{25}\ln\left(\frac{72}{209}\right)
+
\frac{9}{25}\ln\left(\frac{137}{209}\right)
\right]
\approx 0.8,
```
as in any validation loss that is less than 0.8 will be considered to be better than guessing.

Reference

Karpathy, A., Johnson, J., Yeung, S. (2016). CS231n: Convolutional Neural Networks for Visual Recognition. Stanford University Course Notes. [cs231](cs231n.github.io/convolutional-networks/).

Springenberg, J. T., et al (2014). Striving for simplicity: The all convolutional net. [arxiv](arxiv.org/abs/1412.6806).

Szegedy, Christian, et. al (2014). Going Deeper with Convolutions. [arxiv](arxiv.org/abs/1409.4842).


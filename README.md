# Cloud Patterns
In this project a convolutional neural network was trained to classify cloud formation patterns in satellite images. The data[1], provided by the Max Planck Institute for Meteorology, consists of over 5,000 labeled satellite images. The set of labels consists of four defined cloud patterns which are named Fish, Flower, Gravel, and Sugar. 

<img src = examples/labels.jpg width = 700>

The learning task took the form of a multi-label classification problem, as a full satellite image can have multiple patterns present.


### Model and Data Augmentation
I used the ResNet34[2] model with pre-trained weights for the learning task. The ResNet models were originally trained with images that were transformed to a size of 224x224 (resizing and random cropping). The raw satellite images had a size of 2100x1400. In order to prevent distortion when resizing the image, I created a transformation that padded the height of the image to a size  of 2100x2100 and then resized the image to 224x224, which maintained the aspect ratio. For the training process several transformations were applied to the images including random rotations, random vertical/horizontal flips, grayscaling, normalizing etc. 


### Training 
The last layer (fully connected linear layer) of the ResNet model was removed and replaced with a new linear layer with four outputs corresponding to the labels. Training was split into two phases. Initially, only the weights of the linear layer were being updated and the rest of the network weights(pre-trained) were frozen to allow the model to adjust to the new learning task. This phase lasted for 4 epochs. After 4 epochs, all of model weights were unfrozen and the model was fine-tuned using a smaller learning rate. 
Without this freezing process, pre-trained weights could be updated in a way that negate the benefits of transfer learning such as useful representations learned from the previous training task.


The data was split according to a 60 20 20 ratio (test,validation,train). The hyperparameter that was tuned was the 'number of epochs'. After each epoch of the training loop, the loss on the validation set was computed and stored.  I implemented early stopping in order to terminate the training loop if the validation loss did not decrease after a specified number of epochs (patience). The patience was set to 15 epochs.

### Summary
| | |
| ----------- | ----------- |
| Model    | ResNet34 w/ modified linear layer (4 outputs)     |
| Learning Rate  | Phase 1: `.001`, Phase 2: `.00001`  (Adam optimizer) |
| Batch Size |  `32` |

## Results

<img src = plots/loss.png  width = 475>

The optimal number of epochs was determined to be 19 (the training loop stopped after 35 epochs). A new model was then initialized and trained for 19 epochs on the combined train and validation set. To evaluate model performance on the test set, I computed three metrics:

| Metric  | Values |
| ----------- | ----------- |
| Partial Accuracy  (1 - Hamming Loss) | 0.752  |
| Per-label accuracy  |  [0.718, 0.841, 0.711, 0.738]   |
| Exact Accuracy |  0.317|

### Further work
The pre-trained weights in the latter layers of the ResNet34 model could be less relevant to the current task of processing satellite images. I would want to explore how different fine-tuning and freezing strategies affect the validation loss. Additionally, the dataset provided pixels corresponding to each label for each image. I would re-formulate the learning task as a pixel-wise classification using the U-net architecture (semantic segmentation).


## Loss Function
A data point is represented as $(x_i,y_{1i},y_{2i},y_{3i},y_{4i})$ where, `Fish = 1, Flower = 2, Gravel = 3, Sugar = 4` and $x_i$ is the 'i-th' image. The distribution of an individual label conditioned on the corresponding image is modeled as a bernoulli distribution:

$$ 
P(y_{ci} \ | \ x_i) = \text{Bernoulli}(p_{ci}) = p_{ci}^{y_{ci}}*(1-p_{ci})^{(1-y_{ci})}
$$

$$ p_{ci} = \sigma(\text{model}(x_i;w)_c)
$$

where $w$ denotes the weights of neural network and $\sigma$ denotes the sigmoid function. The distribution of all the labels conditioned on the corresponding image is modeled as a multinomial distribution:

$$ P(y_{1i},y_{2i},y_{3i},y_{4i} | x_i) = \prod_{c=1}^4 p_{ci}^{y_{ci}}*(1-p_{ci})^{(1-y_{ci})}$$ 

We assume that the dataset was generated from some distribution. If we assume that the data points are independent of each other then the conditional-likelihood of the data can be expressed as:

$$ 
\prod_{i=1}^{N} P(y_{1i},y_{2i},y_{3i},y_{4i} | x_i) = \prod_{i=1}^{N}\prod_{c=1}^4 p_{ci}^{y_{ci}}*(1-p_{ci})^{(1-y_{ci})}
$$ 

The model weights that maximize the likelihood of the data or equivalently minimize the negative log likelihood of the data are denoted as:

$$
\begin{align}
\arg\min_w  -\log(\prod_{i=1}^{N}\prod_{c=1}^4 p_{ci}^{y_{ci}}*(1-p_{ci})^{(1-y_{ci})}) \\
\arg\min_w \sum_{i=1}^{N}\sum_{c=1}^4 -y_{ci}*log(p_{ci})-(1-y_{ci})*log(1-p_{ci})
\end{align} 
$$


The pytorch functions `torch.nn.BCEWithLogitsLoss` and `torch.nn.MultiLabelSoftMarginLoss` both implement the following loss function:


$$ 
\text{Loss}(w) = \frac{1}{(N*C)}\sum_{i=1}^{N}\sum_{c=1}^C -y_{ci}*log(p_{ci})-(1-y_{ci})*log(1-p_{ci})
$$


```python
def loss_implementation(logits,labels):
    'logits = model(image_batch)'
    batch_size, num_labels, = labels.shape
    param1 = torch.sigmoid(logits)
    param2 = 1 - torch.sigmoid(logits)
    term1 = -labels*torch.log(param1)
    term2 = -(1-labels)*torch.log(param2)
    values = term1 + term2
    double_sum = torch.sum(values)
    loss = (1/(batch_size*num_labels))*double_sum
    return loss
```

# References
1. https://www.kaggle.com/competitions/understanding_cloud_organization
2. https://pytorch.org/vision/main/models/resnet.html



# Saliency Methods

## Introduction

This repository contains code for the following saliency techniques:

*   XRAI* ([paper](https://arxiv.org/abs/1906.02825), [poster](https://github.com/PAIR-code/saliency/blob/master/docs/ICCV_XRAI_Poster.pdf))
*   SmoothGrad* ([paper](https://arxiv.org/abs/1706.03825))
*   Vanilla Gradients
    ([paper](https://scholar.google.com/scholar?q=Visualizing+higher-layer+features+of+a+deep+network&btnG=&hl=en&as_sdt=0%2C22),
    [paper](https://arxiv.org/abs/1312.6034))
*   Guided Backpropogation ([paper](https://arxiv.org/abs/1412.6806))
*   Integrated Gradients ([paper](https://arxiv.org/abs/1703.01365))
*   Occlusion
*   Grad-CAM ([paper](https://arxiv.org/abs/1610.02391))
*   Blur IG ([paper](https://arxiv.org/abs/2004.03383))

And in this repository, you use to evaluate a blastocyst by them.

If you try it in easy, you should [this example iPython notebook](https://github.com/kamikami1107/Invisible-Cradle/blob/i-c/Examples.ipynb)
showing these techniques is a good starting place.

And you try them more, you should [this example iPython notebook](https://github.com/kamikami1107/Invisible-Cradle/blob/i-c/Examples_more.ipynb) 
shows
these techniques is a good starting place.


## Download

```
# To install the core subpackage:
pip install saliency

# To install core and tf1 subpackages:
pip install saliency[tf1]

```

or for the development version:
```
git clone https://github.com/pair-code/saliency
cd saliency
```


## Usage

The saliency library has two subpackages:
*	`core` uses a generic `call_model_function` which can be used with any ML 
	framework.
*	`tf1` accepts input/output tensors directly, and sets up the necessary 
	graph operations for each method.



Each saliency mask class extends from the `CoreSaliency` base class. This class
contains the following methods:

*   `GetMask(x_value, call_model_function, call_model_args=None)`: Returns a mask
    of
    the shape of non-batched `x_value` given by the saliency technique.
*   `GetSmoothedMask(x_value, call_model_function, call_model_args=None, stdev_spread=.15, nsamples=25, magnitude=True)`: 
    Returns a mask smoothed of the shape of non-batched `x_value` with the 
    SmoothGrad technique.


The visualization module contains two methods for saliency visualization:

* ```VisualizeImageGrayscale(image_3d, percentile)```: Marginalizes across the
  absolute value of each channel to create a 2D single channel image, and clips
  the image at the given percentile of the distribution. This method returns a
  2D tensor normalized between 0 to 1.
* ```VisualizeImageDiverging(image_3d, percentile)```: Marginalizes across the
  value of each channel to create a 2D single channel image, and clips the
  image at the given percentile of the distribution. This method returns a
  2D tensor normalized between -1 to 1 where zero remains unchanged.

If the sign of the value given by the saliency mask is not important, then use
```VisualizeImageGrayscale```, otherwise use ```VisualizeImageDiverging```. See
the SmoothGrad paper for more details on which visualization method to use.

##### call_model_function
`call_model_function` is how we pass inputs to a given model and receive the outputs
necessary to compute saliency masks. The description of this method and expected 
output format is in the `CoreSaliency` description, as well as separately for each method.



Here is a condensed example of using IG+SmoothGrad with TensorFlow 2:

```
import saliency.core as saliency
import tensorflow as tf

...

# call_model_function construction here.
def call_model_function(x_value_batched, call_model_args, expected_keys):
	tape = tf.GradientTape()
	grads = np.array(tape.gradient(output_layer, images))
	return {saliency.INPUT_OUTPUT_GRADIENTS: grads}

...

# Load data.
image = GetImagePNG(...)

# Compute IG+SmoothGrad.
ig_saliency = saliency.IntegratedGradients()
smoothgrad_ig = ig_saliency.GetSmoothedMask(image, 
											call_model_function, 
                                            call_model_args=None)

# Compute a 2D tensor for visualization.
grayscale_visualization = saliency.VisualizeImageGrayscale(
    smoothgrad_ig)
```

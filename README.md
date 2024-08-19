<div align="center">
  <img src="images/meta_icon.jpg" alt="Algorithm icon">
  <h1 align="center">infer_segment_anything_2</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_segment_anything_2">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_segment_anything_2">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_segment_anything_2/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_segment_anything_2.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

This algorithm proposes inference for the Segment Anything Model 2 (SAM2). It can be used to generate masks for all objects in an image. With its promptable segmentation capability, SAM delivers unmatched versatility for various image analysis tasks. 

![Sam2 car](https://raw.githubusercontent.com/Ikomia-hub/infer_segment_anything_2/main/images/output_auto.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow
```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo  = wf.add_task(name="infer_segment_anything_2", auto_connect=True)

# Run directly on your image
wf.run_on(url="https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/images/cars.jpg?raw=true")

# Inspect your result
display(algo.get_image_with_mask())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
### General parameters
- `model_name` `[str]` : The SAM model can be loaded with four different encoders:
    - sam2_hiera_tiny - size: 38.9
    - sam2_hiera_small - size: 46
    - sam2_hiera_base_plus - size: 80.8
    - sam2_hiera_large - size: 224.4
- `cuda` `[bool]`: If True, CUDA-based inference (GPU). If False, run on CPU.
- `input_size_percent` `[int]` : Percentage size of the input image. Can be reduce to save memory usage. 

### Prompt predictor parameters
- `input_box` `[list]`: A Nx4 array of given box prompts to the  model, in [[XYXY]] or [[XYXY], [XYXY]] format.
- `input_point` `[list]`: A Nx2 array of point prompts to the model. Each point is in [[X,Y]] or [[XY], [XY]] in pixels.
- `input_point_label` `[list]`: A length N array of labels for the point prompts. 1 indicates a foreground point and 0 indicates a background point
- `multimask_output` - `[bool]`: if true, the model will return three masks. If only a single mask is needed, the model's predicted quality score can be used to select the best mask. For non-ambiguous prompts, such as multiple input prompts, multimask_output=False can give better results.


### Automatic predictor parameters
- `points_per_side` `[int or None]` - the number of points to be sampled
along one side of the image. The total number of points is
- `points_per_batch` - `[int]` - sets the number of points run simultaneously
by the model. Higher numbers may be faster but use more GPU memory.
- `iou_thresh` `[float]` - a filtering threshold in `[0,1]`, using the
model's predicted mask quality.
- `stability_score_thresh` - `[float]` - a filtering threshold in `[0,1]`, using
the stability of the mask under changes to the cutoff used to binarize
the model's mask predictions.
- `stability_score_offset` - `[float]` - the amount to shift the cutoff when
calculated the stability score.
- `box_nms_thresh` - `[float]` - the box IoU cutoff used by non-maximal
suppression to filter duplicate masks.
- `crop_n_layers` - `[int]` - if `>0`, mask prediction will be run again on
crops of the image. Sets the number of layers to run, where each
layer has `2`i_layer` number of image crops.
- `crop_nms_thresh` - `[float]` - the box IoU cutoff used by non-maximal
suppression to filter duplicate masks between different crops.
- `crop_overlap_ratio` - `[float]` - sets the degree to which crops overlap.
In the first crop layer, crops will overlap by this fraction of
the image length. Later layers with more crops scale down this overlap.
- `crop_n_points_downscale_factor` - `[int]` - the number of points-per-side
sampled in layer `n` is scaled down by `crop_n_points_downscale_factor`n`.
- `use_m2m` - `[bool]`: Whether to add a one step refinement using previous mask predictions.


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo  = wf.add_task(name="infer_segment_anything_2", auto_connect=True)

# Example: Setting parameters for 'prompt prediction':
algo.set_parameters({
    # "points_per_side": "32", # For 'automatic predictor'
    "input_size_percent": "100",
    "input_point":"[[500,500]]",
    "multimask_output":"True" # Generate 3 output masks
})

# Run directly on your image
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_cat.jpg?raw=true")

# Inspect your result
img_output = algo.get_output(0)
mask_output = algo.get_output(1)
mask_output_2 = algo.get_output(2)
mask_output_3 = algo.get_output(3)
# display(img_output.get_image_with_mask(mask_output), title="Mask 1")
display(img_output.get_image_with_mask(mask_output_2), title="Mask 2")
# display(img_output.get_image_with_mask(mask_output_3), title="Mask 3")
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo  = wf.add_task(name="infer_segment_anything_2", auto_connect=True)

# Run directly on your image
wf.run_on(url="https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/images/cars.jpg?raw=true")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

## :fast_forward: Advanced usage 

## 1. Automated mask generation
When no prompt is used, SAM2 will generate masks automatically over the entire image. 
You can select the number of masks using the parameter "Points per side" on Ikomia STUDIO or "points_per_side" with the API. 

![Sam dog auto](https://raw.githubusercontent.com/Ikomia-hub/infer_segment_anything/main/images/dog_auto_seg.png)

## 2. Segmentation mask with graphic prompts:
Given a graphic prompts: a single point or boxes SAM can predict masks over the desired objects. 
- Ikomia API: 
    - Using prompt coordinate
        - Point: 'input_point' parameter, e.g. [xy]
        - Point label: 'input_point_label' parameters, e.g. [1,0] 1 to include, 0 to exclude from mask
        - Box: 'input_box' parameter, e,g, [xyxy] or [[xyxy], [xyxy]].
- Ikomia STUDIO:
    - Using graphics
        - Point: Select the point tool
        - Box: Select the Square/Rectangle tool
    - Using coordinate prompts
        - Point: 'Point coord. xy (optional)' [xy]
        - Point label: [1,0], 1 to include, 0 to exclude from mask
        - Box: 'Box coord. xyxy (optional)' [[xyxy], [xyxy]]

### a. Single point 
SAM2 with `"multimask_output":"True"` generate three outputs given a single point (3 best scores). 
![Sam dog single](https://raw.githubusercontent.com/Ikomia-hub/infer_segment_anything/main/images/dog_single_point.png)


### b. Multiple points
A single point can be ambiguous, using multiple points can improve the quality of the expected mask.

### c. Boxes
Drawing a box over the desired object usually output a mask closer to expectation compared to point(s). 

SAM can also take multiple inputs prompts.
![Sam cat boxes](https://raw.githubusercontent.com/Ikomia-hub/infer_segment_anything/main/images/cats_boxes.png)

### d. Point and box

Point and box can be combined by including both types of prompts to the predictor. Here this can be used to select just the trucks's tire, instead of the entire wheel.
![truck_box_point](https://raw.githubusercontent.com/Ikomia-hub/infer_segment_anything/main/images/truck_box_point.png)

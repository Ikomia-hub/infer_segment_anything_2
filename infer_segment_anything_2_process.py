import copy
import torch
import numpy as np
import cv2
import json
import hydra

from ikomia import core, dataprocess, utils

from infer_segment_anything_2.utils_ik import *
from infer_segment_anything_2.sam_2.sam2.build_sam import build_sam2
from infer_segment_anything_2.sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor
from infer_segment_anything_2.sam_2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from omegaconf import DictConfig

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferSegmentAnything2Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "sam2_hiera_small"
        self.apply_postprocessing = False
        self.points_per_side = 32
        self.points_per_batch = 64
        self.stability_score_thresh = 0.95
        self.stability_score_offset = 1.0
        self.box_nms_thresh = 0.7
        self.mask_threshold = 0.0
        self.iou_thres = 0.8
        self.crop_n_layers = 0
        self.crop_nms_thresh = 0.70
        self.crop_overlap_ratio = float(512 / 1500)
        self.crop_n_points_downscale_factor = 1
        self.input_size_percent = 100
        self.input_point = ''
        self.input_box = ''
        self.input_point_label = ''
        self.cuda = torch.cuda.is_available()
        self.use_m2m = True
        self.multimask_output = False
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.apply_postprocessing = utils.strtobool(param_map["apply_postprocessing"])
        self.points_per_side = int(param_map["points_per_side"])
        self.points_per_batch = int(param_map["points_per_batch"])
        self.iou_thres = float(param_map["iou_thres"])
        self.stability_score_thresh = float(param_map["stability_score_thresh"])
        self.stability_score_offset = float(param_map["stability_score_offset"])
        self.box_nms_thresh = float(param_map["box_nms_thresh"])
        self.mask_threshold = float(param_map["mask_threshold"])
        self.crop_n_layers = int(param_map["crop_n_layers"])
        self.crop_nms_thresh = float(param_map["crop_nms_thresh"])
        self.crop_overlap_ratio = float(param_map["crop_overlap_ratio"])
        self.crop_n_points_downscale_factor = int(param_map["crop_n_points_downscale_factor"])
        self.input_size_percent = int(param_map["input_size_percent"])
        self.input_point = param_map['input_point']
        self.input_point_label = param_map['input_point_label']
        self.input_box = param_map['input_box']
        self.use_m2m = utils.strtobool(param_map["use_m2m"])
        self.multimask_output = utils.strtobool(param_map["multimask_output"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["apply_postprocessing"] = str(self.apply_postprocessing)
        param_map["points_per_side"] = str(self.points_per_side)
        param_map["points_per_batch"] = str(self.points_per_batch)
        param_map["iou_thres"] = str(self.iou_thres)
        param_map["stability_score_thresh"] = str(self.stability_score_thresh)
        param_map["stability_score_offset"] = str(self.stability_score_offset)
        param_map["box_nms_thresh"] = str(self.box_nms_thresh)
        param_map["mask_threshold"] = str(self.mask_threshold)
        param_map["crop_n_layers"] = str(self.crop_n_layers)
        param_map["crop_overlap_ratio"] = str(self.crop_overlap_ratio)
        param_map["crop_nms_thresh"] = str(self.crop_nms_thresh)
        param_map["crop_n_points_downscale_factor"] = str(self.crop_n_points_downscale_factor)
        param_map["input_size_percent"] = str(self.input_size_percent)
        param_map["input_point"] = str(self.input_point)
        param_map["input_point_label"] = str(self.input_point_label)
        param_map["input_box"] = str(self.input_box)
        param_map["use_m2m"] = str(self.use_m2m)
        param_map["multimask_output"] = str(self.multimask_output)
        param_map["cuda"] = str(self.cuda)
        return param_map


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferSegmentAnything2(dataprocess.CSemanticSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CSemanticSegmentationTask.__init__(self, name)
        # Add input/output of the algorithm here
        # Create parameters object
        if param is None:
            self.set_param_object(InferSegmentAnything2Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
        self.device = torch.device("cuda")
        self.sam2_model = None
        self.mask_generator = None
        self.predictor = None
        self.input_point = None
        self.input_label = np.array([1]) # foreground point
        self.input_box = None
        self.dtype = torch.float32

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def infer_mask_generator(self, image):
        # Generate mask
        results = self.mask_generator.generate(image)

        if len(results) > 0:
            mask_output = np.zeros_like(results[0]["segmentation"], dtype=np.int8)

            for i, mask_bool in enumerate(results):
                mask_output += mask_bool["segmentation"] * (i + 1)

        else:
            print("No mask predicted, increasing the number of points per side may help")
            mask_output = None
        return [mask_output]

    def infer_predictor(self, graph_input, src_image, resizing, param):
        self.input_box = None
        self.input_label = None
        self.input_point = None
        # Get input from coordinate prompt param - STUDIO/API
        if param.input_box or param.input_point:
            if param.input_box:
                box_list = json.loads(param.input_box)
                self.input_box = np.array(box_list)
                self.input_box = self.input_box * resizing
                self.input_label = np.array([0]) # background point

            if param.input_point:
                point = json.loads(param.input_point)
                self.input_point = np.array(point)
                self.input_point = self.input_point * resizing

        # Get input from drawn graphics - STUDIO
        else:
            graphics = graph_input.get_items() #Get list of input graphics items.
            box = []
            point = []
            for i, graphic in enumerate(graphics):
                bboxes = graphics[i].get_bounding_rect() # Get graphic coordinates
                if graphic.get_type() == core.GraphicsItem.RECTANGLE: # rectangle
                    x1 = bboxes[0]*resizing
                    y1 = bboxes[1]*resizing
                    x2 = (bboxes[2]+bboxes[0])*resizing
                    y2 = (bboxes[3]+bboxes[1])*resizing
                    box.append([x1, y1, x2, y2])
                    self.input_box = np.array(box)
                    self.input_label = np.array([0]) # background point
                if graphic.get_type() == core.GraphicsItem.POINT: # point
                    x1 = bboxes[0]*resizing
                    y1 = bboxes[1]*resizing
                    point.append([x1, y1])
                    self.input_point = np.array(point)

        # Calculate the necessary image embedding
        self.predictor.set_image(src_image)

        # Inference from multiple boxes
        if self.input_box is not None and len(self.input_box) > 1:
            if self.input_point is not None:
                    print('Point input(s) not used, please select a correct graphic input combination')
            masks, _, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box,
                        multimask_output=param.multimask_output,
                        )
            mask_output = np.zeros((
                            src_image.shape[0],
                            src_image.shape[1]
                            ))

            masks = np.squeeze(masks)
            for i, mask_bool in enumerate(masks):
                mask = mask_bool
                i += 1
                mask_output = mask_output + mask * i

        # Inference from points
        elif self.input_point is not None and self.input_box is None:
            if len(self.input_point) == 1:
                masks, _, _ = self.predictor.predict(
                    point_coords=self.input_point,
                    point_labels=self.input_label,
                    multimask_output=param.multimask_output,
                )

            if len(self.input_point) > 1:
                if param.input_point_label:
                    self.input_label = json.loads(param.input_point_label)
                    self.input_label = np.array(self.input_label)
                    # Edit input label if the user makes a mistake
                    if len(self.input_label) != len(self.input_point):
                        self.input_label = np.ones(len(self.input_point))
                else:
                    # Automatically generate input labels
                    self.input_label = np.ones(len(self.input_point))

                masks, _, _ = self.predictor.predict(
                    point_coords=self.input_point,
                    point_labels=self.input_label,
                    multimask_output=param.multimask_output,
                )

        # Inference from a single box
        elif self.input_point is None and len(self.input_box) == 1:
            masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=self.input_box[None, :],
            multimask_output=param.multimask_output,
            )

        # Inference from a single box and a single point
        elif self.input_point is not None and len(self.input_box) == 1:
            if len(self.input_point) > 1:
                self.input_box = None
                print('Box input(s) not used, please select a correct graphic input combination')
                if param.input_point_label:
                    self.input_label = json.loads(param.input_point_label)
                    self.input_label = np.array(self.input_label)
                    # Edit input label if the user makes a mistake
                if len(self.input_label) != len(self.input_point):
                    self.input_label = np.ones(len(self.input_point))
            else:
                self.input_label = np.array([0])

            masks, _, _ = self.predictor.predict(
                point_coords=self.input_point,
                point_labels=self.input_label,
                box=self.input_box,
                multimask_output=param.multimask_output,
            )

        else:
            masks = np.zeros([src_image.shape[:2]])
            print("Please select a point and/or a box")

        return masks

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Get input (numpy array)
        src_image = self.get_input(0).get_image()

        # Check the number of channels
        if src_image.shape[-1] == 4:  # RGBA?
            src_image = src_image[:, :, :3]  # Keep only RGB channels

        # Resize image
        ratio = param.input_size_percent / 100
        h_orig, w_orig = src_image.shape[0], src_image.shape[1]
        if param.input_size_percent < 100:
            width = int(src_image.shape[1] * ratio)
            height = int(src_image.shape[0] * ratio)
            dim = (width, height)
            src_image = cv2.resize(src_image, dim, interpolation = cv2.INTER_LINEAR)

        # Load model
        if param.update or self.sam2_model is None:
            # Check for float16 and bfloat16 support
            float16_support, bfloat16_support = check_float16_and_bfloat16_support()
            # Determine dtype based on GPU support
            self.dtype = torch.bfloat16 if bfloat16_support else torch.float16 \
                            if float16_support else torch.float32
            self.device = torch.device("cuda") if param.cuda and \
                            torch.cuda.is_available() else torch.device("cpu")
            checkpoint, config_folder, model_cfg = get_model(param.model_name)

            # Clear existing Hydra instance
            hydra.core.global_hydra.GlobalHydra.instance().clear()

            # Reinitialize Hydra with the new configuration module path
            hydra.initialize_config_dir(config_dir=config_folder, job_name="infer_segment_anything_2")

            # Load the configuration
            cfg = hydra.compose(config_name=model_cfg)

            # Ensure cfg is a valid configuration object
            if not isinstance(cfg, DictConfig):
                raise TypeError("Configuration is not a valid DictConfig object")

            self.sam2_model = build_sam2(
                                model_cfg,
                                checkpoint,
                                device=self.device,
                                apply_postprocessing=param.apply_postprocessing)
            param.update = False

        # Check graphic input prompt
        graph_input = self.get_input(1)
        if graph_input.is_data_available() or param.input_box or param.input_point:
            if self.predictor is None:
                self.predictor = SAM2ImagePredictor(self.sam2_model)
            with torch.autocast(device_type="cuda" if param.cuda else "cpu", dtype=self.dtype):
                masks = self.infer_predictor(
                                    graph_input=graph_input,
                                    src_image=src_image,
                                    resizing=ratio,
                                    param=param
                )

        else:
            self.mask_generator = SAM2AutomaticMaskGenerator(
                                    model=self.sam2_model,
                                    points_per_side=param.points_per_side,
                                    points_per_batch=param.points_per_batch,
                                    pred_iou_thresh=param.iou_thres,
                                    stability_score_thresh=param.stability_score_thresh,
                                    stability_score_offset=param.stability_score_offset,
                                    mask_threshold=param.mask_threshold,
                                    box_nms_thresh=param.box_nms_thresh,
                                    crop_n_layers=param.crop_n_layers,
                                    crop_overlap_ratio=param.crop_overlap_ratio,
                                    crop_nms_thresh=param.crop_nms_thresh,
                                    crop_n_points_downscale_factor= param.crop_n_points_downscale_factor,
                                    use_m2m=param.use_m2m,
                                    multimask_output=param.multimask_output
            )
            with torch.autocast(device_type="cuda" if param.cuda else "cpu", dtype=self.dtype):
                masks = self.infer_mask_generator(src_image)

        # Set image output
        if len(masks) > 1:
            # self.remove_output(1)
            for i, mask in enumerate(masks):
                self.add_output(dataprocess.CSemanticSegmentationIO())
                mask = mask.astype("uint8")
                if param.input_size_percent < 100:
                    mask = resize_mask(mask, h_orig, w_orig)
                output = self.get_output(i+1)
                output.set_mask(mask)

        else:
            mask = masks[0].astype("uint8")
            if param.input_size_percent < 100:
                mask = resize_mask(mask, h_orig, w_orig)
            # Set output mask (Semantic Seg)
            self.get_output(0)
            self.set_mask(mask)

        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferSegmentAnything2Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        # Set process information as string here
        self.info.name = "infer_segment_anything_2"
        self.info.short_description = "Inference for Segment Anything Model 2 (SAM2)."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/meta_icon.jpg"
        self.info.authors = "Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, " \
                            "Haitham and Radle, Roman and Rolland, Chloe and Gustafson, "  \
                            "Laura and Mintun, Eric and Pan, Junting and Alwala, " \
                            "Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan " \
                            "and Girshick, Ross and Dollar, Piotr and Feichtenhofer, Christoph"
        self.info.article = "SAM 2: Segment Anything in Images and Videos"
        self.info.journal = "ArXiv"
        self.info.year = 2024
        self.info.license = "Apache 2.0 license"
        # URL of documentation
        self.info.documentation_link = "https://ai.meta.com/blog/segment-anything-2/"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_segment_anything_2"
        self.info.original_repository = "https://github.com/facebookresearch/segment-anything-2"
        # Python version
        self.info.min_python_version = "3.11.0"
        # Keywords used for search
        self.info.keywords = "SAM, ViT, Zero-Shot, SA-V dataset, Meta"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "SEMANTIC_SEGMENTATION"

    def create(self, param=None):
        # Create algorithm object
        return InferSegmentAnything2(self.info.name, param)

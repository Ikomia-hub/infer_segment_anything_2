from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_segment_anything_2.infer_segment_anything_2_process import InferSegmentAnything2Param
from torch.cuda import is_available
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferSegmentAnything2Widget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferSegmentAnything2Param()
        else:
            self.parameters = param

        # Create layout : QVBoxLayout by default
        self.grid_layout = QVBoxLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Section 1: Always Visible
        self.section1 = QGroupBox("General Settings")
        self.section1_layout = QGridLayout()
        self.section1.setLayout(self.section1_layout)
        self.grid_layout.addWidget(self.section1)

        self.check_cuda = pyqtutils.append_check(self.section1_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        self.combo_model_name = pyqtutils.append_combo(self.section1_layout, "Model name")
        self.combo_model_name.addItem("sam2_hiera_tiny")
        self.combo_model_name.addItem("sam2_hiera_small")
        self.combo_model_name.addItem("sam2_hiera_base_plus")
        self.combo_model_name.addItem("sam2_hiera_large")

        self.combo_model_name.setCurrentText(self.parameters.model_name)

        self.spin_input_size_percent = pyqtutils.append_spin(self.section1_layout,
                                                             "Image size (%)",
                                                             self.parameters.input_size_percent,
                                                             min=1, max=100)

        # Section 2: PROMPT PREDICTOR
        self.toggle_prompt_button = QPushButton("Toggle PROMPT PREDICTOR")
        self.toggle_prompt_button.setCheckable(True)
        self.toggle_prompt_button.setChecked(False)
        self.toggle_prompt_button.toggled.connect(self.toggle_prompt_group)
        self.grid_layout.addWidget(self.toggle_prompt_button)

        self.prompt_group = QGroupBox("PROMPT PREDICTOR")
        self.prompt_layout = QGridLayout()
        self.prompt_group.setLayout(self.prompt_layout)
        self.prompt_group.setVisible(False)
        self.grid_layout.addWidget(self.prompt_group)

        self.check_multimask_output = pyqtutils.append_check(self.prompt_layout, "Multimask output", self.parameters.multimask_output)

        self.edit_box_input = pyqtutils.append_edit(self.prompt_layout,
                                                    "Box coord. [[xyxy]]",
                                                    self.parameters.input_box
                                                    )
        
        self.edit_point_input = pyqtutils.append_edit(self.prompt_layout,
                                            "Point coord. [[xy]]",
                                            self.parameters.input_point
                                            )
        
        self.edit_point_label = pyqtutils.append_edit(self.prompt_layout,
                                    "Point label [i]",
                                    self.parameters.input_point_label
                                    )

        # Section 2: AUTOMATIC MASK GENERATOR
        self.toggle_automask_button = QPushButton("Toggle AUTOMATIC MASK GENERATOR")
        self.toggle_automask_button.setCheckable(True)
        self.toggle_automask_button.setChecked(False)
        self.toggle_automask_button.toggled.connect(self.toggle_automask_group)
        self.grid_layout.addWidget(self.toggle_automask_button)

        self.automask_group = QGroupBox("AUTOMATIC MASK GENERATOR")
        self.automask_layout = QGridLayout()
        self.automask_group.setLayout(self.automask_layout)
        self.automask_group.setVisible(False)
        self.grid_layout.addWidget(self.automask_group)

        self.spin_points_per_side = pyqtutils.append_spin(self.automask_layout,
                                                          "Points per side",
                                                          self.parameters.points_per_side,
                                                          min=1)
        self.spin_points_per_batch = pyqtutils.append_spin(self.automask_layout,
                                                           "Points per batch",
                                                           self.parameters.points_per_batch,
                                                           min=1)
        self.spin_iou_thres = pyqtutils.append_double_spin(self.automask_layout,
                                                           "IOU threshold",
                                                           self.parameters.iou_thres,
                                                           min=0.0, max=1.0, step=0.01)
        self.spin_stability_score_thresh = pyqtutils.append_double_spin(self.automask_layout,
                                                                        "Stability score threshold",
                                                                        self.parameters.stability_score_thresh,
                                                                        min=0.0, max=1.0, step=0.01)
        self.spin_stability_score_offset = pyqtutils.append_double_spin(self.automask_layout,
                                                                        "Stability score offset",
                                                                        self.parameters.stability_score_offset,
                                                                        min=0.0, max=1.0, step=0.01)
        self.spin_box_nms_thresh = pyqtutils.append_double_spin(self.automask_layout,
                                                                "Box NMS threshold",
                                                                self.parameters.box_nms_thresh,
                                                                min=0.0, max=1.0, step=0.01)
        self.spin_mask_threshold = pyqtutils.append_double_spin(self.automask_layout,
                                                                "Mask threshold",
                                                                self.parameters.mask_threshold,
                                                                min=0.0, max=1.0, step=0.01)
        self.spin_crop_n_layers = pyqtutils.append_spin(self.automask_layout,
                                                        "Crop N layers",
                                                        self.parameters.crop_n_layers,
                                                        min=1)
        self.spin_crop_nms_thresh = pyqtutils.append_double_spin(self.automask_layout,
                                                                 "Crop NMS threshold",
                                                                 self.parameters.crop_nms_thresh,
                                                                 min=0.0, max=1.0, step=0.01)
        self.spin_crop_overlap_ratio = pyqtutils.append_double_spin(self.automask_layout,
                                                                    "Crop overlap ratio",
                                                                    self.parameters.crop_overlap_ratio,
                                                                    min=0.0, max=1.0, step=0.01)
        self.spin_crop_n_points_downscale_factor = pyqtutils.append_spin(self.automask_layout,
                                                                         "Crop points downscale factor",
                                                                         self.parameters.crop_n_points_downscale_factor,
                                                                         min=1)

        self.check_use_m2m = pyqtutils.append_check(self.automask_layout, "Use M2M", self.parameters.use_m2m)

        # Set widget layout
        self.set_layout(layout_ptr)

    def toggle_automask_group(self, checked):
        self.automask_group.setVisible(checked)

    def toggle_prompt_group(self, checked):
        self.prompt_group.setVisible(checked)

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.model_name = self.combo_model_name.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.points_per_side = self.spin_points_per_side.value()
        self.parameters.points_per_batch = self.spin_points_per_batch.value()
        self.parameters.iou_thres = self.spin_iou_thres.value()
        self.parameters.stability_score_thresh = self.spin_stability_score_thresh.value()
        self.parameters.stability_score_offset = self.spin_stability_score_offset.value()
        self.parameters.box_nms_thresh = self.spin_box_nms_thresh.value()
        self.parameters.mask_threshold = self.spin_mask_threshold.value()
        self.parameters.crop_n_layers = self.spin_crop_n_layers.value()
        self.parameters.crop_nms_thresh = self.spin_crop_nms_thresh.value()
        self.parameters.crop_overlap_ratio = self.spin_crop_overlap_ratio.value()
        self.parameters.crop_n_points_downscale_factor = self.spin_crop_n_points_downscale_factor.value()
        self.parameters.input_size_percent = self.spin_input_size_percent.value()
        self.parameters.input_box = self.edit_box_input.text()
        self.parameters.input_point = self.edit_point_input.text()
        self.parameters.input_point_label = self.edit_point_label.text()
        self.parameters.use_m2m = self.check_use_m2m.isChecked()
        self.parameters.multimask_output = self.check_multimask_output.isChecked()
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferSegmentAnything2WidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_segment_anything_2"

    def create(self, param):
        # Create widget object
        return InferSegmentAnything2Widget(param, None)

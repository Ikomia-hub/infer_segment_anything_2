from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_segment_anything_2.infer_segment_anything_2_process import InferSegmentAnything2Factory
        return InferSegmentAnything2Factory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_segment_anything_2.infer_segment_anything_2_widget import InferSegmentAnything2WidgetFactory
        return InferSegmentAnything2WidgetFactory()

import cntk as C
import json
from processing import ImageProcessor
from modelhublib.model import ModelBase


class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # load the DL model
        self._model = C.Function.load("model/model.onnx", device=C.device.cpu(), format=C.ModelFormat.ONNX)


    def infer(self, input):
        # load preprocessed input
        inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)
        # run inference
        results = self._model.eval({self._model.arguments[0]:[inputAsNpArr]})
        # postprocess results into output
        output = self._imageProcessor.computeOutput(results)
        return output

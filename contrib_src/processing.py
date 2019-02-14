from modelhublib.processor import ImageProcessorBase
import PIL
import numpy as np
import json
import cntk as C

class ImageProcessor(ImageProcessorBase):

    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):
            # model input shape
            input_shape = (1, 1, 64, 64)
            # load image and resize -
            img = image.resize((64, 64), PIL.Image.ANTIALIAS) #
            # convert to numpy array (64x64x3) or (64x64x4) - ignoring single channel images.
            img_arr = np.array(img)
            # new arr
            new_arr = np.empty([64,64])
            # if 3 or 4 channels
            if len(img_arr.shape) > 2:
                if img_arr.shape[2] == 3 or img_arr.shape[2] == 4 :
                    for i in range(64):
                      for j in range(64):
                          rgb = img_arr[i][j]
                          new_arr[i][j] = (rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114  - 127.5)/127.5
            else:
                for i in range(64):
                  for j in range(64):
                      new_arr[i][j] = (img_arr[i][j]  - 127.5)/127.5
            input = np.resize(new_arr, input_shape)
        else:
            raise IOError("Image Type not supported for preprocessing.")
        return input

    def _preprocessAfterConversionToNumpy(self, npArr):
        # pass
        return npArr

    def computeOutput(self, inferenceResults):
        probs = C.softmax(inferenceResults).eval()
        probs = np.squeeze(np.asarray(probs))
        with open("model/labels.json") as jsonFile:
            labels = json.load(jsonFile)
        result = []
        for i in range (len(probs)):
            obj = {'label': str(labels[str(i)]),
                    'probability': float(probs[i])}
            result.append(obj)
        return result

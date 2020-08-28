import sys

sys.path.append('..')
from eval_kit.detector import DeeperForensicsDetector

class ToyPredictor(DeeperForensicsDetector):
    def __init__(self):
        super(ToyPredictor, self).__init__()

    def predict(self, video_frames):
        # Here, we just simply return possibility of 0.5
        return 0.5

"""
This script provides a local test routine so you can verify the algorithm works before pushing it to evaluation.

It runs your detector on several local videos and verify whether they have obvious issues, e.g:
    - Fail to start
    - Wrong output format

It also prints out the runtime for the algorithms for your references.


The participants are expected to implement a face forgery detector class. The sample detector illustrates the interface.
Do not modify other part of the evaluation toolkit otherwise the evaluation will fail.

DeeperForensics Challenge
"""

import time
import sys
import logging

from eval_kit.client import get_local_frames_iter, verify_local_output

logging.basicConfig(level=logging.INFO)

sys.path.append('model')
########################################################################################################
# please change these lines to include your own face forgery detector extending the eval_kit.detector.DeeperForensicsDetector base class.
from toy_predict import ToyPredictor as DeeperForensicsDetector
########################################################################################################


def run_local_test(detector_class, frames_iter):
    """
    In this function we create the detector instance. And evaluate the wall time for performing DeeperForensicsDetector.
    """

    # initialize the detector
    logging.info("Initializing face detector.")
    try:
        detector = detector_class()
    except:
        # send errors to the eval frontend
        raise
    logging.info("Detector initialized.")


    # run the images one-by-one and get runtime
    output_probs = {}
    output_times = {}
    num_frames = {}
    eval_cnt = 0

    logging.info("Starting runtime evaluation")
    for video_id, frames in frames_iter:
        time_before = time.time()
        try:
            prob = detector.predict(frames)
            assert isinstance(prob, float)
            num_frames[video_id] = len(frames)
            output_probs[video_id] = prob
        except:
            # send errors to the eval frontend
            logging.error("Video id failed: {}".format(video_id))
            raise
        elapsed = time.time() - time_before
        output_times[video_id] = elapsed
        logging.info("video {} run time: {}".format(video_id, elapsed))

        eval_cnt += 1

        if eval_cnt % 100 == 0:
            logging.info("Finished {} videos".format(eval_cnt))

    logging.info("""
    ================================================================================
    all videos finished, showing verification info below:
    ================================================================================
    """)

    # verify the algorithm output
    verify_local_output(output_probs, output_times, num_frames)


if __name__ == '__main__':
    local_test_video_iter = get_local_frames_iter()
    run_local_test(DeeperForensicsDetector, local_test_video_iter)

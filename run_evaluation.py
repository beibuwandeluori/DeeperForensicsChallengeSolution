"""
The evaluation entry point for DeeperForensics Challenge.

It will be the entrypoint for the evaluation docker once built.
Basically It downloads a list of videos and run the detector on each video.
Then the runtime output will be reported to the evaluation system.

The participants are expected to implement a Deepfakes detector class. The sample detector illustrates the interface.
Do not modify other part of the evaluation toolkit otherwise the evaluation will fail.

DeeperForensics Challenge
"""
import time
import sys
import logging

from eval_kit.client import upload_eval_output, get_frames_iter, get_job_name


logging.basicConfig(level=logging.INFO)

sys.path.append('model')
########################################################################################################
# please change these lines to include your own face forgery detector extending the eval_kit.detector.DeeperForensicsDetector base class.
from toy_predict import ToyPredictor as DeeperForensicsDetector
########################################################################################################


def evaluate_runtime(start_time, detector_class, frames_iter, job_name):
    """
    Please DO NOT modify this part of code or the eval_kit
    Modification of the evaluation toolkit could result in cancellation of your award.

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

    logging.info("all videos finished, uploading evaluation outputs for evaluation.")
    total_time = time.time() - start_time
    # send evaluation output to the server
    upload_eval_output(output_probs, output_times, num_frames, job_name, total_time)


if __name__ == '__main__':
    job_name = get_job_name()
    start_time = time.time()
    deeper_forensics_frames_iter = get_frames_iter()
    evaluate_runtime(start_time, DeeperForensicsDetector, deeper_forensics_frames_iter, job_name)

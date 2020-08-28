def extract_frames(video_path):
    """
    Extract frames from a video. You can use either provided method here or implement your own method.

    params:
        - video_local_path (str): the path of video.
    return:
        - frames (list): a list containing frames extracted from the video.
    """
    ########################################################################################################
    # You can change the lines below to implement your own frame extracting method (and possibly other preprocessing),
    # or just use the provided codes.
    import cv2
    vid = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = vid.read()
        if not success:
            break
        if frame is not None:
            frames.append(frame)
        # Here, we extract one frame only without other preprocessing
        if len(frames) >= 1:
            break
    vid.release()
    return frames
    ########################################################################################################

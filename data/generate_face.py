import numpy as np
import cv2
import os
import detect_face
import shutil
import tensorflow as tf
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_boundingbox(bb, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = bb[0]
    y1 = bb[1]
    x2 = np.minimum(bb[2], width)
    y2 = np.minimum(bb[3], height)
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    scale_x1 = max(int(center_x - size_bb // 2), 0)
    scale_y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - scale_x1, size_bb)
    size_bb = min(height - scale_y1, size_bb)

    x1 = abs(scale_x1 - x1)
    x2 = abs(x2 - scale_x1)
    y1 = abs(scale_y1 - y1)
    y2 = abs(y2 - scale_y1)

    return scale_x1, scale_y1, size_bb, x1, x2, y1, y2


def extract_frames(videos_path, frame_subsample_count=30, output_path=None,
                   pnet=None, rnet=None, onet=None):
    reader = cv2.VideoCapture(videos_path)
    # fps = video.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    while reader.isOpened():
        success, whole_image = reader.read()
        if not success:
            break
        if frame_num % frame_subsample_count == 0:
            cropped_face = get_face(whole_image, pnet=pnet, rnet=rnet, onet=onet)
            if cropped_face == []:
                continue
            # plt.imshow(cropped_face)
            # plt.show()
            save_path = os.path.join(output_path, '{:04d}.png'.format(frame_num))
            cv2.imwrite(save_path, cropped_face)
        frame_num += 1
    reader.release()


def get_face(whole_image, pnet=None, rnet=None, onet=None):
    # Face detector
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Image size
    height, width = whole_image.shape[:2]
    image = cv2.cvtColor(whole_image, cv2.COLOR_BGR2RGB)

    bounding_boxes, points = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    cropped_face = []

    if bounding_boxes.shape[0]:
        # For now only take biggest face
        prob_index = np.argsort(bounding_boxes[:, 4])
        bb = bounding_boxes[prob_index[-1]][0:4].astype(np.int32)
        scale_x1, scale_y1, size_bb, x1, x2, y1, y2 = \
            get_boundingbox(bb, width, height, scale=1.2)
        cropped_face = whole_image[scale_y1:scale_y1 + size_bb,
                       scale_x1:scale_x1 + size_bb]
        # cropped_face = whole_image[bb[1]:bb[3],
        #                bb[0]:bb[2]]
    return cropped_face


def main():
    # video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/end_to_end'
    # face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/end_to_end'

    # video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/end_to_end_level_1'
    # face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/end_to_end_level_1'

    # video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/end_to_end_level_2'
    # face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/end_to_end_level_2'

    # video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/end_to_end_level_3'
    # face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/end_to_end_level_3'

    # video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/end_to_end_level_4'
    # face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/end_to_end_level_4'

    # video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/end_to_end_level_5'
    # face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/end_to_end_level_5'

    # video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/end_to_end_mix_2_distortions'
    # face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/end_to_end_mix_2_distortions'

    # video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/end_to_end_mix_3_distortions'
    # face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/end_to_end_mix_3_distortions'

    # video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/end_to_end_mix_4_distortions'
    # face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/end_to_end_mix_4_distortions'

    # video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/end_to_end_random_level'
    # face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/end_to_end_random_level'

    video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/reenact_postprocess'
    face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/reenact_postprocess'

    model_path = 'mtcnn_models'
    if not os.path.isdir(face_path):
        os.mkdir(face_path)
        print(face_path)
    start = 0
    end = start+1
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, model_path)

    video_file_path = video_path
    face_file_path = face_path
    if not os.path.isdir(face_file_path):
        os.mkdir(face_file_path)
    for name in tqdm(os.listdir(video_file_path)):
        input_path = os.path.join(video_file_path, name)
        if name.find('.mp4') == -1:
            try:
                shutil.copy(input_path, face_file_path)
                continue
            except:
                continue
        output_path = os.path.join(face_file_path, name)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        if len(os.listdir(output_path)) != 0:
            continue
        extract_frames(input_path, frame_subsample_count=20, output_path=output_path,
                       pnet=pnet, rnet=rnet, onet=onet)


def getFile(path, format='mp4'):
    files = os.listdir(path)  # 得到文件夹下的所有文件，包含文件夹名称
    FileList = []
    for name in files:
        if os.path.isdir(os.path.join(path, name)):
            FileList.extend(getFile(os.path.join(path, name), format)) #回调函数，对所有子文件夹进行搜索
        elif os.path.isfile(os.path.join(path, name)):
            if format.lower() in name.lower():
                FileList.append(os.path.join(path, name))
        else:
            print("未知文件:%s", name)
    return FileList


def main_real(vid):
    video_path = '/data1/cby/dataset/DeepForensic/videos/source_videos/' + vid
    face_path = '/data1/cby/dataset/DeepForensic/face_images/source_images/' + vid

    model_path = 'mtcnn_models'
    if not os.path.isdir(face_path):
        os.mkdir(face_path)
        print(face_path)
    start = 0
    end = start+1
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, model_path)

    video_file_path = video_path
    face_file_path = face_path
    if not os.path.isdir(face_file_path):
        os.mkdir(face_file_path)
    for input_path in tqdm(getFile(video_file_path, format='mp4')):
        # output_path = os.path.join(face_file_path, input_path)
        output_path = input_path.replace(video_path, face_path)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        if len(os.listdir(output_path)) != 0:
            continue
        extract_frames(input_path, frame_subsample_count=50, output_path=output_path,
                       pnet=pnet, rnet=rnet, onet=onet)


if __name__ == "__main__":
    # main_real()
    # video_path = '/data1/cby/dataset/DeepForensic/videos/source_videos/M004'
    # face_path = '/data1/cby/dataset/DeepForensic/face_images/manipulated_images/M004'
    # file_list = getFile(path=video_path, format='mp4')
    # print(file_list[0], '\n', file_list[0].replace(video_path, face_path))
    # print(len(file_list))
    vids = os.listdir('/data1/cby/dataset/DeepForensic/videos/source_videos')
    print('vids total lenght:', len(vids))
    start = 70
    end = start + 10
    print(vids[start:end], start, end)
    for i, vid in enumerate(vids[start:end]):
        print(i, 'Start extract face in', vid)
        main_real(vid)
        print(i, 'Extract face in', vid, 'Finished!')




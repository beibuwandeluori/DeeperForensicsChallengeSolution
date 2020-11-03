import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm

def extract_frames(videos_path, frame_subsample_count=30, output_path=None):
    reader = cv2.VideoCapture(videos_path)
    # fps = video.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    while reader.isOpened():
        success, whole_image = reader.read()
        if not success:
            break
        if frame_num % frame_subsample_count == 0:
            save_path = os.path.join(output_path, '{:04d}.png'.format(frame_num))
            cv2.imwrite(save_path, whole_image)
            frame_num += 1
            break
    reader.release()


def main(vid):
    video_path = '/data1/cby/dataset/DeepForensic/videos/manipulated_videos/' + vid
    face_path = '/data1/cby/dataset/DeepForensic/frames/manipulated_images/' + vid

    if not os.path.isdir(face_path):
        os.mkdir(face_path)
        print(face_path)
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
        extract_frames(input_path, frame_subsample_count=20, output_path=output_path)


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
    face_path = '/data1/cby/dataset/DeepForensic/frames/source_images/' + vid
    if not os.path.isdir(face_path):
        os.mkdir(face_path)
        print(face_path)
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
        extract_frames(input_path, frame_subsample_count=50, output_path=output_path)


if __name__ == "__main__":
    # vids = os.listdir('/data1/cby/dataset/DeepForensic/videos/source_videos')
    # print('vids total lenght:', len(vids))
    # start = 80
    # end = start + 20
    # print(vids[start:end], start, end)
    # for i, vid in enumerate(vids[start:end]):
    #     print(start + i, 'Start extract frames in', vid)
    #     main_real(vid)
    #     print(start + i, 'Extract frames in', vid, 'Finished!')

    vids = os.listdir('/data1/cby/dataset/DeepForensic/videos/manipulated_videos')
    print('vids total lenght:', len(vids))
    start = 9
    end = start + 3
    print(vids[start:end], start, end)
    for i, vid in enumerate(vids[start:end]):
        print(start + i, 'Start extract frames in', vid)
        main(vid)
        print(start + i, 'Extract frames in', vid, 'Finished!')




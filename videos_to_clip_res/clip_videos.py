from moviepy.editor import VideoFileClip
import os
import cv2


# 加载视频
save_dir = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/videos_to_clip_res/'
video_dir_name = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/videos_to_clip/'
video_dir = os.listdir(video_dir_name)

for video_name in video_dir:
    video = video_dir_name + video_name

    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 裁剪区域：左右各去掉50像素
    x1, x2 = 270, w - 330
    out = cv2.VideoWriter(save_dir + video_name, fourcc, fps, (x2 - x1, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[:, x1:x2]  # 裁剪左右
        out.write(cropped_frame)

    cap.release()
    out.release()


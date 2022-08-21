import cv2

def save_img(addr, imag, num):
    path = addr + '/img_' + str(num) + '.jpg'
    cv2.imwrite(path, imag)

video_path = '../../apex_data/video/apex_04.mp4'
img_path = '../../apex_data/img'

video_capture = cv2.VideoCapture(video_path)
open, frame = video_capture.read()
total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

all_frame = True
start_frame = round(1 / 978 * total)         #开始帧率
end_frame = round(56 / 978 * total)          #结束帧率

start_frame_0 = round(69 / 978 * total)         #开始帧率0
end_frame_0 = round(12 / 16 * total)          #结束帧率0

start_frame_1 = round(14 / 16 * total)         #开始帧率1
end_frame_1 = round(16 / 16 * total)       #结束帧率2

time_interval = 20      #间隔帧率
time_interval_1 = 60       #间隔帧率1

i = 0
j = 1383
while open:
    i += 1
    if i % time_interval == 0 :
        if all_frame:
            j += 1
            save_img(img_path, frame, j)
            print(f'------save img {j} sucessfully!------')
        elif start_frame <= i <= end_frame:
                j += 1
                save_img(img_path, frame, j)
                print(f'------save img {j} sucessfully!------')
        elif start_frame_0 <= i <= end_frame_0:
            j += 1
            save_img(img_path, frame, j)
            print(f'------save img {j} sucessfully!------')
    elif i % time_interval_1 == 0:
        if start_frame_1 <= i <= end_frame_1:
            j += 1
            save_img(img_path, frame, j)
            print(f'------save img {j} sucessfully!------')
    open, frame =video_capture.read()

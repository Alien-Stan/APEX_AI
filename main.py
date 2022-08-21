import numpy as np
import win32api

from aim_apex.grabscreen import grab_screen
from aim_apex.apex_model import model
import cv2
import win32con
import win32gui
import torch

from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from aim_apex.img_process import img_porcess
from aim_apex.mouse_lock import lock
import pynput

x, y = (1980, 1080)  #截屏分辨率
x_0, y_0 = (1980, 1080)  #显示分辨率

imgsz = 640
lock_mode = False
weights = r'G:\yolov5\yolov5-6.1\runs\train\exp22\weights\best.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device!= 'cpu'
model = model(weights, imgsz, half)
mouse = pynput.mouse.Controller()

# 超参数
conf_thres = 0.4
iou_thres = 0.45


def on_click(x, y, button, pressed):
    global lock_mode
    if pressed and button == button.x2:
        lock_mode = not lock_mode
        print('on' if lock_mode else 'off')

listener = pynput.mouse.Listener(on_click=on_click)
listener.start()
while True:


# with pynput.mouse.Events() as events:
#     while True:
#         it = next(events)
#         while it is not None and not isinstance(it, pynput.mouse.Events.Click):
#             it = next(events)
#         if it is not None and it.button == it.button.x2 and it.pressed:
#             lock_mode = not lock_mode
#             print('on' if lock_mode else 'off')

    img0 = grab_screen(region=(0, 0 , x, y))
    img0 = cv2.resize(img0, (x_0, y_0))
    img = img_porcess(img0, model, imgsz)

    # 数据处理
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255
    if len(img.shape) == 3:
        img = img[None]

    # 加载模型
    pred = model(img, augment=False, visualize=False)
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    # 预测
    aims = []
    for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        s = ''

        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()
            s += f"{n} {model.names[int(c)]}{'s' * (n > 1)}, "
        for *xyxy, conf, cls in reversed(det):
            '''
            0 enemy_body  1 enemy_head
            '''
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            line = (cls, *xywh)
            aim = ('%g ' * len(line)).rstrip() % line
            aim = aim.split(' ')
            aims.append(aim)


        if lock_mode:
            mouse_x_pos, mouse_y_pos = win32api.GetCursorPos()
            # mouse_x_pos,  mouse_y_pos = mouse.position
            print((mouse_x_pos, mouse_y_pos))
            m = []
            for win in aims:
                _, c_x, c_y, c_w, c_h = win
                area = float(c_w) * float(c_h)
                m.append(area)
            det = aims[np.argmax(m)]

            tag, x_center, y_center, w, h = det
            x_center, w = float(x_center) * x, float(w) * x
            y_center, h = float(y_center) * y, float(h) * y
            print(tag)
            print(x_center, y_center)

            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, round(x_center - mouse_x_pos),
                                 round(y_center - mouse_y_pos))
        for i ,xywh in enumerate(aims):
            _, x_center, y_center, w, h = [float(i) for i in xywh]
            x_center, w = x_center * x_0, w * x_0
            y_center, h = y_center * y_0, h * y_0

            top_left = (round(x_center - w / 2), round(y_center - h / 2))
            buttom_right = (round(x_center + w / 2), round(y_center + h / 2))
            cv2.rectangle(img0, top_left, buttom_right, color=(0, 255, 0), thickness=3)





    cv2.namedWindow('apex_detect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('apex_detect', x_0 // 2, y_0 // 2)
    cv2.imshow('apex_detect', img0)


    hwnd = win32gui.FindWindow(None, 'apex_detect')
    cvrect = cv2.getWindowImageRect('apex_detect')
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    if cv2.waitKey(1) & 0xFF == 27:
        cv2.destroyAllWindows()
        break


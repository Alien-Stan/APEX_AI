import pynput
import numpy as np
import pydirectinput
import win32api
import win32con


def lock(aims, mouse, x, y):
    mouse_x_pos,  mouse_y_pos = win32api.GetCursorPos()
    # mouse_x_pos,  mouse_y_pos = mouse.position
    print((mouse_x_pos,  mouse_y_pos))
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

    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE , round(x_center - mouse_x_pos), round(y_center - mouse_y_pos))
    # win32api.SetCursorPos((200,200))
    # pydirectinput.moveTo(round(x_center), round(y_center - 1 / 5 * h))
    # mouse.position = (x_center, y_center - 1 / 5 * h)
    # if tag == 0:
    #     mouse.move(10,10)
    #     print('body')
    #     mouse.position = (x_center, y_center - 1 / 5 *h)
    # if tag == 1:
    #     print('head')
    #     mouse.position = (x_center, y_center)
    # mouse.position = (x_center, y_center)









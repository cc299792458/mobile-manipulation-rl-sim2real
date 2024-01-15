import multiprocessing
import tkinter as tk
from utils import *

def keyboard_interface(queue):
    """
    创建一个简单的 tkinter 界面来捕捉键盘事件。
    捕捉到的按键通过多进程队列传递。
    """
    def on_key_press(event):
        queue.put(event.char)
    root = tk.Tk()
    root.bind('<KeyPress>', on_key_press)
    root.mainloop()

def main():
    dynamixel_arm = DynamixelArm()
    alpha, r, z, beta, gripper = 0, 0.1, 0.3, 0, 0.03

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=keyboard_interface, args=(queue,))
    process.start()
    try:
        while True:
            if not queue.empty():
                key_pressed = queue.get()
                if key_pressed == 'w':
                    z += 0.005
                elif key_pressed == 's':
                    z -= 0.005
                elif key_pressed == 'a':
                    r += 0.005
                elif key_pressed == 'd':
                    r -= 0.005
                elif key_pressed == 'j':
                    beta += np.pi/36
                elif key_pressed == 'k':
                    beta -= np.pi/36
                elif key_pressed == 'u':
                    alpha += np.pi/36
                elif key_pressed == 'i':
                    alpha -= np.pi/36
                elif key_pressed == 'f':
                    gripper += 0.002
                elif key_pressed == 'g':
                    gripper -= 0.002
                dynamixel_arm.set_ee_pos(ee_pos=[alpha, r, z, beta, gripper])
                dynamixel_arm.get_current(if_print=True)
    except KeyboardInterrupt:
        print("exit")
    finally:
        process.terminate()

if __name__ == "__main__":
    main()
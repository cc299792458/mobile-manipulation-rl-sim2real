import cv2
import numpy as np
from concept.envs import PickCube_v0

class OpenCVViewer:
    def __init__(self, name="OpenCVViewer", is_rgb=True, exit_on_esc=True):
        self.name = name
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        self.is_rgb = is_rgb
        self.exit_on_esc = exit_on_esc

    def imshow(self, image: np.ndarray, is_rgb=None, non_blocking=False, delay=0):
        if image.ndim == 2:
            image = np.tile(image[..., np.newaxis], (1, 1, 3))
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        assert image.ndim == 3, image.shape

        if self.is_rgb or is_rgb:
            image = image[..., ::-1]
        cv2.imshow(self.name, image)

        if non_blocking:
            return
        else:
            key = cv2.waitKey(delay)
            if key == 27:  # escape
                if self.exit_on_esc:
                    exit(0)
                else:
                    return None
            elif key == -1:  # timeout
                pass
            else:
                return chr(key)

    def close(self):
        cv2.destroyWindow(self.name)

    def __del__(self):
        self.close()


if __name__ == '__main__':
    env = PickCube_v0(only_arm=False, controller_type='delta_target_ee_control')
    obs = env.reset()
    opencv_viewer = OpenCVViewer(exit_on_esc=False)
    while True:
        action = np.zeros([7])
        render_frame = env.render(mode='cameras')
        key = opencv_viewer.imshow(render_frame)
        if key == 'q':
            break
        elif key == 'w':
            action[0] += 0.5
            # env.step(np.array([1., 0, 0, 0, 0, 0, 0]))
        elif key == 's':
            action[0] -= 0.5
            # env.step(np.array([-1., 0, 0, 0, 0, 0, 0]))
        elif key == 'a':
            action[1] += 0.5
            # env.step(np.array([0.0, 1., 0, 0, 0, 0, 0]))
        elif key == 'd':
            action[1] -= 0.5
            # env.step(np.array([0.0, -1., 0, 0, 0, 0, 0]))
        elif key == 'j':
            action[3] += 0.5
            # env.step(np.array([0.0, 0, 0.0, 1., 0, 0, 0]))
        elif key == 'k':
            action[4] += 0.5
            # env.step(np.array([0.0, 0, 0.0, -1., 0, 0, 0]))
        elif key == 'l':
            action[5] += 0.5
            # env.step(np.array([0.0, 0, 0.0, 0, 1., 0, 0]))
        elif key == 'u':
            action[3] -= 0.5
            # env.step(np.array([0.0, 0, 0.0, 0, -1., 0, 0]))
        elif key == 'i':
            action[4] -= 0.5
            # env.step(np.array([0.0, 0, 0.0, 0, 0.0, 1., 0]))
        elif key == 'o':
            action[5] -= 0.5
            # env.step(np.array([0.0, 0, 0.0, 0, 0.0, -1., 0]))
        elif key == 'f':
            action[6] += 0.5
            # env.step(np.array([0.0, 0, 0.0, 0, 0.0, 0.0, 1.]))
        elif key == 'g':
            action[6] -= 0.5
            # env.step(np.array([0.0, 0, 0.0, 0, 0.0, 0.0, -1.]))
        elif key == "r":  # reset env
            obs = env.reset()
        obs, rew, done, info = env.step(action=action)
        print(f"reward: {rew}")
        print(f"info: {info}")
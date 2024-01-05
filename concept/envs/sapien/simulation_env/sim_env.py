import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from sapien.utils import Viewer

import gym
from gym import spaces

CONTROLLER_TYPE = ['delta_joint_control', 'delta_ee_control']

class SimEnv(gym.Env):
    def __init__(self,
                 controller_type='delta_joint_control', 
                 render_mode='console',
                 only_arm=True):
        """
            controller_type: controller type, related to action space
            render_mode: human or console
        """
        gym.Env.__init__(self)
        ###### Set up Engine, Renderer, and Viewer
        self.engine, self.renderer = self.set_up_engine_renderer()
        self.scene = self.set_up_scene(self.engine)
        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.viewer = self.set_up_viewer(self.scene, self.renderer)
            self.render_camera = None
        else:
            self.render_camera = self.scene.add_camera('render_camera', width=1024, height=848, fovy=1, near=0.05, far=100)
            self.render_camera.set_pose(Pose(p=np.array([0.2, 1.0, 0.3]), q=np.array([0.707, 0.0, 0.0, -0.707])))
            # self.render_camera.set_pose(Pose(p=np.array([1.0, 0.0, 0.3]), q=np.array([0.0, 0.0, 0.0, 1.0])))
            self.viewer = None
        ###### Set up Robot and its Controller
        self.robot = self.initialize_robot(self.scene)
        self.reset_robot()
        self.initilize_controller(controller_type, only_arm)
        ###### Set up Task
        self.initialize_task()
        ###### Others
        obs = self.reset()
        self.observation_space = self._convert_observation_to_space(obs)
        
        
    ######----- Basic Settings -----#####
    def set_up_engine_renderer(self):
        engine = sapien.Engine()
        #TODO: if add depth sensor, maybe need to be modified
        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)
        return engine, renderer

    def set_up_scene(self, engine: sapien.Engine):
        self.simulation_freq = 500
        scene = engine.create_scene()
        scene.set_timestep(1 / self.simulation_freq)
        scene.add_ground(altitude=0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        return scene

    def set_up_viewer(self, scene: sapien.Scene, renderer: sapien.SapienRenderer):
        viewer = Viewer(renderer)
        viewer.set_scene(scene)
        viewer.set_camera_xyz(x=0.3, y=0.6, z=0.5)
        viewer.set_camera_rpy(r=0, p=-np.pi/12, y=np.pi/2)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        return viewer
    
    ######----- Robot Settings -----#####
    def initialize_robot(self, scene: sapien.Scene):
        # Load robot
        filename = './urdf/robot.urdf'
        loader = scene.create_urdf_loader()
        robot_builder = loader.load_file_as_articulation_builder(filename)
        robot = robot_builder.build(fix_root_link=True)
        robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        # Set control parameters
        # TODO: fine-tune control parameters
        base_control_params = np.array([0, 2e4, 5e2])
        arm_control_params = np.array([2e3, 4e2, 5e2])
        gripper_control_params = np.array([5e3, 1e2, 5e2])
        base_joint_names = ["x_joint", "y_joint", "theta_joint"]
        arm_joint_names = [f"Joint_{i}" for i in range(0, 3)]
        for joint in robot.get_active_joints():
            name = joint.get_name()
            if name in base_joint_names:
                joint.set_drive_property(*(1 * base_control_params), mode="force")
            elif name in arm_joint_names:
                joint.set_drive_property(*(1 * arm_control_params), mode="force")
            else:
                joint.set_drive_property(*(1 * gripper_control_params), mode="force")
        # Color the robot arm's links
        visual_body_colors = [[127,127,127],[  0,120,176],[255,127, 42],[  0,159, 60],[226, 41, 44],[149,105,185],[146, 86, 76]]
        robot_links = robot.get_links()
        body_idx = 0
        for link in robot_links:
            if "Link" not in link.get_name() or "tcp" in link.get_name():
                continue
            for visual in link.get_visual_bodies():
                for geom in visual.get_render_shapes():
                    mat = geom.material
                    color = np.array(visual_body_colors[body_idx]) / 255
                    mat.set_base_color(np.array([*color, 1]))
                    mat.set_specular(0.2)
                    mat.set_roughness(0.7)
                    mat.set_metallic(0.1)
                    geom.set_material(mat)
                    body_idx += 1
                    if body_idx >= len(visual_body_colors):
                        body_idx = 0
        # Cache Links
        self.finger_left_link = [link for link in robot.get_links() if link.get_name() == 'Link_gripper_l'][0]
        self.finger_right_link = [link for link in robot.get_links() if link.get_name() == 'Link_gripper_r'][0]
        self.tcp_link = [link for link in robot.get_links() if link.get_name() == 'Link_tcp'][0]

        return robot
    
    def reset_robot(self):
        # [x, y, theta, arm_dof_0, arm_dof_1, arm_dof_2, arm_dof_3, gripper_l, gripper_r]
        init_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  
        self.set_qpos(init_qpos)
        self.set_drive_target(init_qpos)
        self.set_drive_velocity_target(np.zeros_like(init_qpos))

    def initilize_controller(self, controller_type, only_arm):
        """
            delta_joint_control: control the joint directly, using delta_joint.
            delta_ee_control: control the end effector, using delta action. action_space is: theta, r, z, and phi.
                alpha:
                r:
                z:
                beta:
            base's controller is velocity control: [translation_velocity, rotation_velocity]
        """
        if controller_type not in CONTROLLER_TYPE:
            raise NotImplementedError
        
        self.only_arm = only_arm
        self.control_freq = 20
        self.frame_skip = self.simulation_freq // self.control_freq
        self.controller_type = controller_type
        if not only_arm:
            self.base_action_scale = np.array([0.2, 0.314])   # 0.2 m/s, 18 degree/s 
            base_action_dim = 2
        else:
            base_action_scale = np.array([])
            base_action_dim = 0
        if controller_type == 'delta_joint_control':
            self.arm_action_scale = np.array([(10/180*np.pi), (10/180*np.pi), (10/180*np.pi), (10/180*np.pi), 0.002])
            self.arm_action_limit = self.robot.get_qlimits()[3:]
            arm_action_dim = 5  # 4 arm + 1 gripper
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(base_action_dim + arm_action_dim,), dtype=np.float32)    # 2+4+1
        elif controller_type == 'delta_ee_control':
            raise NotImplementedError
            self.arm_action_scale = np.array([(10/180*np.pi), 0.025, 0.025, (10/180*np.pi), 0.002])
            arm_action_limit = None
            self.action_limit = None
            # self.action_limit = self.robot.get_qlimits()[3:]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(base_action_dim + arm_action_dim,), dtype=np.float32)    # 2+4+1
            
    def forward_kinematics(self):
        arm_qpos = self.qpos[3:]
        alpha = arm_qpos[0]
        r = None
        z = None
        beta = arm_qpos[0] + arm_qpos[1] + arm_qpos[2]

        return alpha, r, z, beta

    def inverse_kinematics(self):
        pass

    @property
    def qpos(self):
        return self.robot.get_qpos()
    
    @property
    def qvel(self):
        return self.robot.get_qvel()

    def set_qpos(self, qpos):
        self.robot.set_qpos(qpos)
    
    def set_drive_target(self, qpos_target):
        self.robot.set_drive_target(qpos_target)
    
    def set_drive_velocity_target(self, qvel_target):
        self.robot.set_drive_velocity_target(qvel_target)

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse = self._get_pairwise_contact_impulse(contacts, self.finger_left_link, actor)
        rimpulse = self._get_pairwise_contact_impulse(contacts, self.finger_right_link, actor)

        # direction to open the gripper
        ldirection = -self.finger_left_link.pose.to_transformation_matrix()[:3, 1]
        rdirection = self.finger_right_link.pose.to_transformation_matrix()[:3, 1]

        # angle between impulse and open direction
        langle = self._compute_angle_between(ldirection, limpulse)
        rangle = self._compute_angle_between(rdirection, rimpulse)

        lflag = (
            np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
        )
        rflag = (
            np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle
        )

        return all([lflag, rflag])
    
    ######----- Task Settings -----#####
    def initialize_task(self):
        pass

    def reset_task(self):
        pass

    #####----- Gym Interface -----##### 
    def reset(self):
        self._elapsed_steps = 0
        self.reset_robot()
        self.reset_task()

        return self.get_obs()

    def step(self, action):
        self.set_action(action)
        self._elapsed_steps += 1

        obs = self.get_obs()
        info = self.get_info(obs=obs)
        reward = self.get_reward(obs=obs, action=action, info=info)
        done = self.get_done(obs=obs, info=info)

        return obs, reward, done, info

    def set_action(self, action):
        if not self.only_arm:
            base_action, arm_action = action[0:2], action[2:]
            # Set mobile base action
            base_action = base_action * self.base_action_scale
            ori = self.qpos[2]  # assume the 3rd DoF stands for orientation
            vel_ego = self._rotate_2d_vec_by_angle(np.hstack([base_action[0], np.array([0])]), ori)
            self.set_drive_velocity_target(np.hstack([vel_ego, base_action[-1], np.zeros_like(self.qpos[3:])]))
        else:
            arm_action = action
        # Set arm action and gripper action
        # TODO: add motion generation here.
        if self.controller_type == 'delta_joint_control':
            arm_action = arm_action * self.arm_action_scale
            cur_qpos = self.qpos[3:]
            delta_qpos = np.hstack([arm_action, arm_action[-1]])
            target_qpos = self._clip_with_bounds(cur_qpos + delta_qpos, self.arm_action_limit)
            target_qpos = np.hstack([np.zeros([3]), target_qpos])
            self.set_drive_target(target_qpos)
            
        elif self.controller_type == 'delta_ee_control':
            raise NotImplementedError
        
        for _ in range(self.frame_skip):
            qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True, external=False)
            self.robot.set_qf(qf)
            self.scene.step()
        self.render(self.render_mode)
            
    def get_obs(self):
        obs_robot = self.get_obs_robot()
        obs_task = self.get_obs_task()
        obs = np.hstack([obs_robot, obs_task])

        return obs
    
    def get_obs_robot(self):
        """
            Get robot's proprioception
        """
        # If use target, should add target into observation
        obs_robot = np.hstack([self.qpos, self.qvel])
        return obs_robot

    def get_obs_task(self):
        return np.array([])
    
    def get_info(self, obs):
        info = dict(elapsed_steps=self._elapsed_steps)
        info.update(self.evaluate())
        return info
    
    def evaluate(self):
        return dict()
    
    def get_reward(self, obs, action, info):
        return None
    
    def get_done(self, obs, info):
        return None

    def render(self, mode='console'):
        self.scene.update_render()
        if mode == 'human':
            self.viewer.render()
        elif mode == "rgb_array":
            if self.render_camera is not None:
                self.render_camera.take_picture()
                rgba = self.render_camera.get_float_texture('Color')
                rgb = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)    
            else:
                rgba = self.viewer.window.get_float_texture('Color')
                rgb = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)
            return rgb
            # images = []
            # images.append(rgb)
            # for camera in self._render_cameras.values():
            #     rgba = camera.get_images(take_picture=True)["Color"]
            #     rgb = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)
            #     images.append(rgb)
            # if len(images) == 1:
            #     return images[0]
            # else:
            #     raise NotImplementedError
            # return tile_images(images)
        elif mode == "cameras":
            if self.viewer is not None or self.render_camera is not None:
                images = self.render("rgb_array")
                self.scene.update_render()
                return images
            # if len(self._render_cameras) > 0:
            #     images = [self.render("rgb_array")]
            # else:
            #     images = []
            

    def close(self):
        pass
    
    #####----- Utils -----#####
    # _clip and scale
    def _clip_with_bounds(self, arr, bounds):
        # Using np.clip to apply the bounds to each element
        clipped_arr = np.array([np.clip(arr[i], bounds[i][0], bounds[i][1]) for i in range(len(arr))])
        return clipped_arr
    
    def _rotate_2d_vec_by_angle(self, vec, theta):
        rot_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        return rot_mat @ vec
    
    def _get_pairwise_contact_impulse(self, contacts, actor0: sapien.ActorBase, actor1: sapien.ActorBase):
        pairwise_contacts = self._get_pairwise_contacts(contacts, actor0, actor1)
        total_impulse = self._compute_total_impulse(pairwise_contacts)
        return total_impulse
    
    def _get_pairwise_contacts(self, contacts, actor0: sapien.ActorBase, actor1: sapien.ActorBase):
        pairwise_contacts = []
        for contact in contacts:
            if contact.actor0 == actor0 and contact.actor1 == actor1:
                pairwise_contacts.append((contact, True))
            elif contact.actor0 == actor1 and contact.actor1 == actor0:
                pairwise_contacts.append((contact, False))
        return pairwise_contacts

    def _compute_total_impulse(self, contact_infos):
        total_impulse = np.zeros(3)
        for contact, flag in contact_infos:
            contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
            # Impulse is applied on the first actor
            total_impulse += contact_impulse * (1 if flag else -1)
        return total_impulse
    
    def _compute_angle_between(self, x1, x2):
        """Compute angle (radian) between two vectors."""
        x1, x2 = self._normalize_vector(x1), self._normalize_vector(x2)
        dot_prod = np.clip(np.dot(x1, x2), -1, 1)
        return np.arccos(dot_prod).item()
    
    def _normalize_vector(self, x, eps=1e-6):
        x = np.asarray(x)
        assert x.ndim == 1, x.ndim
        norm = np.linalg.norm(x)
        return np.zeros_like(x) if norm < eps else (x / norm)
    
    def _convert_observation_to_space(self, observation, prefix=""):
        """Convert observation to OpenAI gym observation space (recursively).
        Modified from `gym.envs.mujoco_env`
        """
        if isinstance(observation, (dict)):
            # CATUION: Explicitly create a list of key-value tuples
            # Otherwise, spaces.Dict will sort keys if a dict is provided
            space = spaces.Dict(
                [
                    (k, self._convert_observation_to_space(v, prefix + "/" + k))
                    for k, v in observation.items()
                ]
            )
        elif isinstance(observation, np.ndarray):
            shape = observation.shape
            dtype = observation.dtype
            low, high = self._get_dtype_bounds(dtype)
            if np.issubdtype(dtype, np.floating):
                low, high = -np.inf, np.inf
            space = spaces.Box(low, high, shape=shape, dtype=dtype)
        elif isinstance(observation, (float, np.float32, np.float64)):
            # logger.debug(f"The observation ({prefix}) is a (float) scalar")
            space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
        elif isinstance(observation, (int, np.int32, np.int64)):
            # logger.debug(f"The observation ({prefix}) is a (integer) scalar")
            space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=int)
        elif isinstance(observation, (bool, np.bool_)):
            # logger.debug(f"The observation ({prefix}) is a (bool) scalar")
            space = spaces.Box(0, 1, shape=[1], dtype=np.bool_)
        else:
            raise NotImplementedError(type(observation), observation)

        return space
    
    def _get_dtype_bounds(self, dtype: np.dtype):
        if np.issubdtype(dtype, np.floating):
            info = np.finfo(dtype)
            return info.min, info.max
        elif np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return info.min, info.max
        elif np.issubdtype(dtype, np.bool_):
            return 0, 1
        else:
            raise TypeError(dtype)


if __name__ == '__main__':
    sim_env = SimEnv(render_mode='human', only_arm=False)
    while True:
        sim_env.step(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
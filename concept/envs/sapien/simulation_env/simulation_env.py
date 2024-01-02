import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from sapien.utils import Viewer

class SimulationEnv:
    def __init__(self, visualize=True):
        self.engine, self.renderer = self.set_up_engine_renderer()
        self.scene = self.set_up_scene(self.engine)
        self.visualize = visualize
        if self.visualize:
            self.viewer = self.set_up_viewer(self.scene, self.renderer)
        self.robot = self.load_robot(self.scene)
        self.initialize_robot()
        self.initilize_controller('delta_joint_control')

    def set_up_engine_renderer(self):
        engine = sapien.Engine()
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
        
    def load_robot(self, scene: sapien.Scene):
        filename = './urdf/robot.urdf'
        loader = scene.create_urdf_loader()
        robot_builder = loader.load_file_as_articulation_builder(filename)
        robot = robot_builder.build(fix_root_link=True)
        robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        arm_control_params = np.array([2e3, 4e2, 5e2])  # This PD is far larger than real to improve stability
        gripper_control_params = np.array([2e3, 1e2, 5e2])
        arm_joint_names = [f"Joint_{i}" for i in range(0, 8)]
        for joint in robot.get_active_joints():
            name = joint.get_name()
            if name in arm_joint_names:
                joint.set_drive_property(*(1 * arm_control_params), mode="force")
            else:
                joint.set_drive_property(*(1 * gripper_control_params), mode="force")
        # color the links here
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

        return robot
    
    def initialize_robot(self):
        init_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.set_qpos(init_qpos)
        self.set_drive_target(init_qpos)
        self.set_drive_velocity_target(np.zeros_like(init_qpos))

    @property
    def qpos(self):
        return self.robot.get_qpos()

    def set_qpos(self, qpos):
        self.robot.set_qpos(qpos)
    
    def set_drive_target(self, qpos_target):
        self.robot.set_drive_target(qpos_target)
    
    def set_drive_velocity_target(self, qvel_target):
        self.robot.set_drive_velocity_target(qvel_target)

    def initilize_controller(self, controller_type):
        """
            delta_joint_control: control the joint directly, using delta_joint.
            delta_ee_control: control the end effector, using delta action. action_space is: theta, r, z, and phi.
                theta:
                r:
                z:
                phi:
        """
        self.control_freq = 20
        self.frame_skip = self.simulation_freq // self.control_freq
        self.controller_type = controller_type
        if controller_type == 'delta_joint_control':
            self.action_scale = np.array([0.1, 0.1, 0.1, 0.1, 0.002, 0.002]) #
            self.action_limit = self.robot.get_qlimits()
        elif controller_type == 'delta_ee_control':
            raise NotImplementedError
    
    def _clip_with_bounds(self, arr, bounds):
        # Using np.clip to apply the bounds to each element
        clipped_arr = np.array([np.clip(arr[i], bounds[i][0], bounds[i][1]) for i in range(len(arr))])
        return clipped_arr

    def set_action(self, action):
        if self.controller_type == 'delta_joint_control':
            action = np.append(action, action[-1])
            cur_qpos = self.qpos
            delta_qpos = action * self.action_scale
            target_qpos = self._clip_with_bounds(cur_qpos + delta_qpos, self.action_limit)
            self.set_drive_target(target_qpos)
            
        elif self.controller_type == 'delta_ee_control':
            raise NotImplementedError


    def step(self, action):
        self.set_action(action)
        for _ in range(self.frame_skip):
            qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True, external=False)
            self.robot.set_qf(qf)
            self.scene.step()
            if self.visualize:
                self.scene.update_render()
                self.viewer.render()
    
if __name__ == '__main__':
    sim_env = SimulationEnv()
    while True:
        sim_env.step(np.array([1, 1, 1, 1, 1]))
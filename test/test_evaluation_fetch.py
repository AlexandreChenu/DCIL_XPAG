#!/usr/bin python -w

import os

import gym
import gym_gfetch

import cv2
import pickle
import pdb

import time

import mujoco_py
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/chenu/Desktop/PhD/github/dcil_xpag/")
from skill_extractor import skills_extractor_Mj
from wrappers.gym_vec_env_mujoco import gym_vec_env

import seaborn
seaborn.set_style("whitegrid")


def load_sim_traj(path):

	with open(path, 'rb') as handle:
		sim_traj = pickle.load(handle)

	return sim_traj

def get_demo(demo_path, verbose=0):
	"""
	Extract demo from pickled file
	"""
	L_sim_states = []

	assert os.path.isfile(demo_path)

	with open(demo_path, "rb") as f:
		demo = pickle.load(f)
		# print("demo.keys() = ", demo.keys())
	for sim_state in demo["sim_states"]:
		L_sim_states.append(sim_state)
	return L_sim_states[:250]

def save_frames_as_video(frames, path, it):

	video_name = path + '/fetch_video_' + str(it) + '.mp4'
	height, width, layers = frames[0].shape
	#resize
	percent = 50
	width = int(frames[0].shape[1] * percent / 100)
	height = int(frames[0].shape[0] * percent / 100)

	video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 50.0, (width, height))

	for frame in frames:
		width = int(frame.shape[1] * percent / 100)
		height = int(frame.shape[0] * percent / 100)
		dim = (width, height)
		resize_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
		cvt_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
		video.write(cvt_frame)

	video.release()
	cv2.destroyAllWindows()

	return

def visu_success_zones(goals, ax):
	"""
	Visualize success zones as sphere of radius eps_success around skill-goals
	"""

	for goal in goals:
		u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

		x = goal[0] + 0.05*np.cos(u)*np.sin(v)
		y = goal[1] + 0.05*np.sin(u)*np.sin(v)
		z = goal[2] + 0.05*np.cos(v)
		ax.plot_wireframe(x, y, z, color="blue", alpha = 0.1)
	return

def save_trajectory_plot(L_positions, env):

	_, eval_env, _ = gym_vec_env('GFetchGoal-v0', 1)

	demo_path = "/Users/chenu/Desktop/PhD/github/dcil_xpag/demos/fetch_convert/6.demo"

	s_extractor = skills_extractor_Mj(demo_path, eval_env, eps_state=0.5)
	# print(s_extractor.skills_sequence[0][0][0].shape)
	goals = [env.project_to_goal_space(obs[0][0]) for obs in s_extractor.skills_sequence]

	max_x = 1.
	min_x = 0.
	min_y = 0.
	max_y = 1.5
	min_z = 0.5
	max_z = 1.2

	fig = plt.figure(figsize=(10, 10))
	ax = plt.axes(projection='3d')
	ax.view_init(azim= -90, elev=20)
	# ax.set_xlim(min_x, max_x)
	# ax.set_ylim(0., max_y)
	ax.set_zlim(min_z, max_z)

	visu_success_zones(goals, ax)

	X_gripper, Y_gripper, Z_gripper = [], [], []
	X_obj0, Y_obj0, Z_obj0 = [], [], []



	for i, position in enumerate(L_positions):
		if i % 2 == 0:
			gripper_pos = position[0]
			obj0_pos = position[1]

			print("gripper_pos = ", gripper_pos)

			X_gripper.append(gripper_pos[0])
			Y_gripper.append(gripper_pos[1])
			Z_gripper.append(gripper_pos[2])

			X_obj0.append(obj0_pos[0])
			Y_obj0.append(obj0_pos[1])
			Z_obj0.append(obj0_pos[2])


			if i > 2:
				ax.cla()
				ax.set_xlim(min_x, max_x)
				ax.set_ylim(0., max_y)
				ax.set_zlim(min_z, max_z)
				visu_success_zones(goals, ax)
				# ax.collections[0].remove()

			ax.plot(X_gripper, Y_gripper, Z_gripper, c="blue")
			ax.plot(X_obj0, Y_obj0, Z_obj0, c="green")
			# ax.plot(X_left_foot, Y_left_foot, Z_left_foot, c="lightsteelblue")
			# ax.plot(X_right_hand, Y_right_hand, Z_right_hand, c="crimson")
			# ax.plot(X_left_hand, Y_left_hand, Z_left_hand, c="crimson")

			plt.draw()
			plt.pause(0.001)

			plt.savefig("/Users/chenu/Desktop/PhD/github/dcil_xpag/test/video/fetch/" + str(i) + ".png")




if (__name__=='__main__'):

	env = gym.make("GFetchGoal-v0")

	# traj_path = "/Users/chenu/Desktop/PhD/results_JZ/results_RF_DCIL/fetch/VC/RF_DCIL_1000_0_0.002_True_20221214_1168769/sim_traj_500000.pickle" #498000.pickle"
	# traj_path = "/Users/chenu/Desktop/PhD/results_JZ/DCIL_I_Fetch_1_20230303_2006248/sim_traj_158000.pickle"
	traj_path = "/Users/chenu/Desktop/PhD/results_JZ/DCIL_I_Fetch_1_20230305_503526/sim_traj_495000.pickle"
	# traj_path = "/Users/chenu/Desktop/PhD/results_JZ/DCIL_I_Fetch_1_20230303_689376/sim_traj_194000.pickle"

	# traj_path = "/Users/chenu/Desktop/PhD/results_JZ/results_RF_DCIL/fetch/VC/RF_DCIL_v4_0_0.0005_True_20221210_190863/sim_traj_295000.pickle"
	# demo_path = "/Users/chenu/Desktop/PhD/github/dcil_xpag/demos/humanoid_standup_convert/1.demo"

	sim_traj = load_sim_traj(traj_path)
	# sim_traj = get_demo(demo_path)
	print(len(sim_traj))

	# sim_traj2 = copy.deepcopy(sim_traj)
	# sim_traj += sim_traj2

	frames = []

	env.reset()
	# frame = env.env.sim.render(width=1000, height=1000, camera_name='track')
	# frames.append(frame)
	# time.sleep(0.1)
	# viewer = mujoco_py.MjViewer(env.sim)
	# target = 'gripper'

	L_body = ["gripper_link", "obj0"]
	L_positions = []

	for sim_state in sim_traj[:200]:

		positions = []

		# time.sleep(0.1)
		# print("sim_state = ", sim_state)
		env.set_inner_state(sim_state)
		env.sim.forward()

		for body in L_body:
			body_id = env.sim.model.body_name2id(body)
			positions.append(env.sim.data.body_xpos[body_id].copy())

		L_positions.append(positions)

		# env.env.sim.render(width=1000, height=1000, mode="window")
		# env.env.render()

		# frame = env.env.sim.render(mode="offscreen", width=1080, height=1080)
		# print(env.env.envs[0])
		frame = env.env.render()
		# frame = env.env.render(mode="offscreen", width=1080, height=1080)
		frames.append(frame)

	env.close()

	## save video
	save_frames_as_video(frames, "/Users/chenu/Desktop/PhD/github/dcil_xpag/test/video/", None)

	## save trajectory dynamic plot
	save_trajectory_plot(L_positions, env)

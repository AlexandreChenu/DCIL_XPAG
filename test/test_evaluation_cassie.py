#!/usr/bin python -w

import os

import gym
import gym_cassie_run

import cv2
import pickle
import pdb
import numpy as np
import time

import mujoco_py

import seaborn

import matplotlib.pyplot as plt

import mujoco as mj

from gym.wrappers.monitoring.video_recorder import *

import mujoco_py
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/chenu/Desktop/PhD/github/dcil_xpag/")
from skill_extractor import skills_extractor_Mj
from wrappers.gym_vec_env_mujoco import gym_vec_env

def visu_success_zones(goals, ax):
	"""
	Visualize success zones as sphere of radius eps_success around skill-goals
	"""

	for goal in goals:
		u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:20j]

		x = goal[0] + 0.05*np.cos(u)*np.sin(v)
		y = goal[1] + 0.05*np.sin(u)*np.sin(v)
		z = goal[2] + 0.05*np.cos(v)
		ax.plot_wireframe(x, y, z, color="blue", alpha = 0.1)
	return


def save_trajectory_plot(L_positions, env):

	_, eval_env, _ = gym_vec_env('GCassieGoal-v0', 1)

	demo_path = "/Users/chenu/Desktop/PhD/github/dcil_xpag/demos/cassie_convert/1.demo"

	s_extractor = skills_extractor_Mj(demo_path, eval_env, eps_state=0.5)
	# print(s_extractor.skills_sequence[0][0][0].shape)
	goals = [eval_env.envs[0].unwrapped.project_to_goal_space(obs[0][0]) for i,obs in enumerate(s_extractor.skills_sequence) if i%3==0]

	fig = plt.figure(figsize=(30, 10))
	ax = plt.axes(projection='3d')
	ax.view_init(azim=-65, elev=20)
	ax.set_xlim(-1., 13.)
	ax.set_ylim(-0.35, 0.1)
	ax.set_zlim(0., 1.)

	visu_success_zones(goals, ax)

	X_torso, Y_torso, Z_torso = [], [], []
	X_right_foot, Y_right_foot, Z_right_foot = [], [], []
	X_left_foot, Y_left_foot, Z_left_foot = [], [], []


	for i, position in enumerate(L_positions):
		if i % 2 == 0:
			torso_pos = position[0]
			right_foot_pos = position[1]
			left_foot_pos = position[2]

			print("torso_pos = ", torso_pos)

			X_torso.append(torso_pos[0])
			Y_torso.append(torso_pos[1])
			Z_torso.append(torso_pos[2])

			X_right_foot.append(right_foot_pos[0])
			Y_right_foot.append(right_foot_pos[1])
			Z_right_foot.append(right_foot_pos[2])

			X_left_foot.append(left_foot_pos[0])
			Y_left_foot.append(left_foot_pos[1])
			Z_left_foot.append(left_foot_pos[2])

			if i > 2:
				ax.cla()
				ax.set_xlim(-1., 13.)
				ax.set_ylim(-0.35, 0.1)
				ax.set_zlim(0., 1.)
				visu_success_zones(goals, ax)
				# ax.collections[0].remove()

			ax.plot(X_torso, Y_torso, Z_torso, c="blue")
			ax.plot(X_right_foot, Y_right_foot, Z_right_foot, c="crimson")
			ax.plot(X_left_foot, Y_left_foot, Z_left_foot, c="crimson")

			# plt.draw()
			# plt.pause(0.001)

			plt.savefig("/Users/chenu/Desktop/PhD/github/dcil_xpag/test/video/cassie_run/" + str(i) + ".png")





def load_sim_traj(path):

	with open(path, 'rb') as handle:
		sim_traj = pickle.load(handle)

	return sim_traj

def load_demo_traj(path):

	with open(path, 'rb') as handle:
		demo = pickle.load(handle)

	sim_traj = demo["sim_states"]

	return sim_traj


if (__name__=='__main__'):

	env = gym.make("CassieRun-v0", render_mode="rgb_array")
	print("env = ", env)
	print("env.__dict__.keys() = ", env.unwrapped.__dict__.keys())
	# print("env.unwrapped.model.__dict__.keys() = ", env.unwrapped.model.__dict__.keys())
	# env.unwrapped.render_mode = "human"
	# traj_path = "/Users/chenu/Desktop/PhD/results_JZ/preliminary_results/preliminary_results_RF_DCIL_humanoid/DCIL_v4_0.4_True_20220725_1515860/sim_traj_" + str(train_it) + ".pickle" #498000.pickle"
	# traj_path = "/Users/chenu/Desktop/PhD/results_JZ/results_DCIL_TRO/DCIL_cassie/DCIL_v4_0.75_True_20221012_2206211/sim_traj_1513000.pickle"
	traj_path = "/Users/chenu/Desktop/PhD/results_JZ/results_DCIL_TRO/supplementary_results/cassie_run/DCIL_v4_0.5_True_20221114_526775/sim_traj_2077000.pickle"
	sim_traj = load_sim_traj(traj_path)
	demo_path = "/Users/chenu/Desktop/PhD/github/DCIL_XPAG/demos/cassie_convert/1.demo"
	demo_traj = load_demo_traj(demo_path)[:len(sim_traj)]

	print(len(sim_traj))

	frames = []

	video_env = VideoRecorder(
	  env,
	  path='/Users/chenu/Desktop/PhD/github/dcil_xpag/test/video/demo_cassie_run.mp4',
	)

	video_env.frames_per_sec = video_env.env.metadata.get("render_fps", 60)

	video_env.env.reset()


	L_body = ["cassie-pelvis", "left_foot", "right_foot"]
	L_colors = ["m", "steelblue", "gray"]
	L_positions = []

	L_positions_demo = []

	# sim_traj = sim_traj + sim_traj
	# sim_traj = [sim_traj[0] for _ in range(10)] + sim_traj

	for sim_state in sim_traj:

		positions = []

		# time.sleep(0.025)
		# print("sim_state = ", sim_state)
		# print("sim_state[0].shape = ", sim_state[0].shape)
		# print("sim_state[1].shape = ", sim_state[1].shape)
		video_env.env.set_inner_state(sim_state)


		# for body in L_body:
		# 	print("body = ", env.unwrapped.data.body(body))
		positions.append(env.unwrapped.data.body("cassie-pelvis").xpos.copy())
		positions.append(env.unwrapped.data.body("left-foot").xpos.copy())
		positions.append(env.unwrapped.data.body("right-foot").xpos.copy())

		print("positions = ", positions)

		L_positions.append(positions)

		video_env.capture_frame()

		# frame = env.env.sim.render(width=3000, height=1000, mode="offscreen")#, camera_id=0)
		# frame = env.render()#, camera_id=0)
		# print("frame = ", frame)
		# time.sleep(0.5)
		# frames.append(frame)

	# video_env.env.reset()

	# for sim_state in demo_traj:
	#
	# 	positions = []
	#
	# 	# time.sleep(0.025)
	# 	# print("sim_state = ", sim_state)
	# 	# print("sim_state[0].shape = ", sim_state[0].shape)
	# 	# print("sim_state[1].shape = ", sim_state[1].shape)
	# 	video_env.env.set_inner_state(sim_state)
	#
	#
	# 	# for body in L_body:
	# 	# 	print("body = ", env.unwrapped.data.body(body))
	# 	positions.append(env.unwrapped.data.body("cassie-pelvis").xpos.copy())
	# 	positions.append(env.unwrapped.data.body("left-foot").xpos.copy())
	# 	positions.append(env.unwrapped.data.body("right-foot").xpos.copy())
	#
	# 	print("positions = ", positions)
	#
	# 	L_positions_demo.append(positions)

	# video_env.env.close()
	video_env.close()

	# fig = plt.figure()
	# ax = fig.add_subplot(projection='3d')
	# # ax.set_aspect('equal')
	# ax.grid(False)
	#
	# # for body_indx, positions in enumerate(L_positions):
	# for body_indx,(_, color) in enumerate(zip(L_body,L_colors)):
	# 	X = [pos[body_indx][0] for pos in L_positions]
	# 	Y = [pos[body_indx][1] for pos in L_positions]
	# 	Z = [pos[body_indx][2] for pos in L_positions]
	#
	# 	X_demo = [pos[body_indx][0] for pos in L_positions_demo]
	# 	Y_demo = [pos[body_indx][1] for pos in L_positions_demo]
	# 	Z_demo = [pos[body_indx][2] for pos in L_positions_demo]
	# #
	# # 	# print("X = ", X)
	# 	# ax.plot(X,Y,Z, label = L_body[body_indx] , color=color)
	# 	ax.plot(X_demo,Y_demo,Z_demo, label = L_body[body_indx] , color=color)
	#
	# ax.set_box_aspect((np.ptp(X)/4, np.ptp(Y)*2, np.ptp(Z)/2))
	#
	# ax.set_xlim((-1.,14.))
	# ax.set_ylim((-0.4, 0.4))
	# ax.set_zlim((0., 1.3))
	# plt.legend()
	# plt.show()
	#
	# # fig, ax = plt.subplots()
	# # ax.grid(False)
	# #
	# # ax.plot(X,Z, label = L_body[body_indx] + "_eval")
	# # # ax.set_box_aspect((np.ptp(X), np.ptp(Z)))
	# #
	# # ax.set_xlim((-1.,10.))
	# # ax.set_ylim((0., 1.5))
	# # plt.legend()
	# # plt.show()



	# save_frames_as_video(frames, "/Users/chenu/Desktop/PhD/github/dcil_xpag/test/", train_it)

	## save trajectory dynamic plot
	save_trajectory_plot(L_positions, env)

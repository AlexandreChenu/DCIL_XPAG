#!/usr/bin python -w

import os

import gym
import gym_ghumanoid

import cv2
import pickle
import pdb

import copy

import time
import numpy as np

import mujoco_py

import matplotlib.pyplot as plt

from mujoco_py import MjSimState


def load_sim_traj(path):

	with open(path, 'rb') as handle:
		sim_traj = pickle.load(handle)

	return sim_traj

def load_demo_traj(path):
	qpos = np.load(path + "episode/qpos.npy")
	qvel = np.load(path + "episode/qvel.npy")
	actions = np.load(path + "episode/actions.npy")

	return qpos, qvel, actions

def save_frames_as_video(frames, path, it):

	video_name = path + '/video_' + str(it) + '.mp4'
	height, width, layers = frames[0].shape
	#resize
	percent = 50
	width = int(frames[0].shape[1] * percent / 100)
	height = int(frames[0].shape[0] * percent / 100)

	video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

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


if (__name__=='__main__'):

	env = gym.make("HumanoidStandup-v2")

	# traj_path = "/Users/chenu/Desktop/PhD/github/rf_dcil_xpag/demos/humanoid_standup/"
	traj_path = "/Users/chenu/Desktop/PhD/results_JZ/results_humanoid/SAC_humanoid_standup_20220825_56789/"
	qpos, qvel, actions = load_demo_traj(traj_path)

	## Extract body positions from evaluation traj
	L_body = ["right_lower_arm", "left_lower_arm", "left_foot", "right_foot", "torso"]
	# L_body = ["left_foot", "right_foot", "torso"]
	L_positions_eval = []

	demo = {}
	demo["observations"] = []
	demo["sim_states"] = []
	demo["actions"] = []

	frames = []

	env.reset()
	# for i in range(qpos.shape[0]):
	for i in range(75):

		positions = {}

		state = MjSimState(time=0.,
				  qpos = qpos[i],
				  qvel = qvel[i],
				  act = None,
				  udd_state = {})

		demo["sim_states"].append(copy.deepcopy(state))
		demo["actions"].append(copy.deepcopy(actions[i]))

		env.sim.set_state(state)
		env.sim.forward()

		## get full obs (including x,y position of torso)
		data = env.sim.data
		obs = np.concatenate(
			[
				data.qpos.flat,
				data.qvel.flat,
				data.cinert.flat,
				data.cvel.flat,
				data.qfrc_actuator.flat,
				data.cfrc_ext.flat,
			])

		demo["observations"].append(obs.copy())

		frame = env.env.sim.render(width=3000, height=1000, mode="offscreen")
		frames.append(frame)

		for body in L_body:
			# print("\n Body = ", body)
			body_id = env.sim.model.body_name2id(body)
			positions[body] = env.sim.data.body_xpos[body_id].copy()
			# print("pos = ", positions[body])

		L_positions_eval.append(positions)

	env.close()

	save_frames_as_video(frames, "/Users/chenu/Desktop/PhD/github/dcil_xpag/test/", "demo4")

	with open("/Users/chenu/Desktop/PhD/github/dcil_xpag/demos/humanoid_standup_convert/1.demo", 'wb') as handle:
		pickle.dump(demo, handle, protocol=pickle.HIGHEST_PROTOCOL)

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	for body in L_body:
		X = [pos[body][0] for pos in L_positions_eval]
		Y = [pos[body][1] for pos in L_positions_eval]
		Z = [pos[body][2] for pos in L_positions_eval]
		print("X = ", X)

		ax.plot(X,Y,Z, label = body + "_eval")

	plt.legend()
	plt.show()

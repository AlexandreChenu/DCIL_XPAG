#!/usr/bin python -w

import os

import gym
import gym_ghumanoid_standup

import cv2
import pickle
import pdb

import time

import mujoco_py
import copy


def load_sim_traj(path):

	with open(path, 'rb') as handle:
		sim_traj = pickle.load(handle)

	return sim_traj

def save_frames_as_video(frames, path, it):

	video_name = path + '/yoyoyo_video_' + str(it) + '.mp4'
	height, width, layers = frames[0].shape
	#resize
	percent = 100
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

	env = gym.make("GHumanoidGoal_standup-v0")


	for train_it in range(170000, 172000, 2000):
		traj_path = "/Users/chenu/Desktop/PhD/results_JZ/preliminary_results/preliminary_results_DCIL_humanoid/humanoid_standup/DCIL_v4_0.35_True_20220719_191477/sim_traj_" + str(train_it) + ".pickle" #498000.pickle"
		sim_traj = load_sim_traj(traj_path)
		print(len(sim_traj))

		sim_traj2 = copy.deepcopy(sim_traj)

		sim_traj += sim_traj2

		frames = []

		env.reset()
		env.env.sim.render(width=1000, height=1000, mode="window")
		time.sleep(0.1)
		# viewer = mujoco_py.MjViewer(env.sim)
		# target = 'torso'

		L_body = ["right_lower_arm", "left_lower_arm", "left_foot", "right_foot", "torso"]
		L_positions = []

		for sim_state in sim_traj:

			positions = []

			time.sleep(0.1)
			# print("sim_state = ", sim_state)
			env.set_inner_state(sim_state)
			env.sim.forward()

			for body in L_body:
				body_id = env.sim.model.body_name2id(body)
				positions.append(env.sim.data.body_xpos[body_id])

			L_positions.append(positions)

			env.env.sim.render(width=1000, height=1000, mode="window")


			# frame = env.env.sim.render(width=1000, height=1000, mode="offscreen")
			# frames.append(frame)

		env.close()

		# save_frames_as_video(frames, "/Users/chenu/Desktop/PhD/github/dcil_xpag/test/", train_it)

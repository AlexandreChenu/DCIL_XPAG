#!/usr/bin python -w

import os

import gym
import gym_ghumanoid

import cv2
import pickle
import pdb

import time

import mujoco_py


def load_sim_traj(path):

	with open(path, 'rb') as handle:
		sim_traj = pickle.load(handle)

	return sim_traj

def save_frames_as_video(frames, path, it):

	video_name = path + '/video_' + str(it) + '.mp4'
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

	env = gym.make("GHumanoid-v0")





	for train_it in range(938000, 942000, 2000):
		traj_path = "/Users/chenu/Desktop/PhD/results_JZ/preliminary_results/preliminary_results_DCIL_humanoid/v4/DCIL_v4_20220706_1957521/sim_traj_" + str(train_it) + ".pickle" #498000.pickle"
		sim_traj = load_sim_traj(traj_path)
		print(len(sim_traj))

		frames = []

		env.reset()

		# viewer = mujoco_py.MjViewer(env.sim)
		# target = 'torso'

		for sim_state in sim_traj:
			# time.sleep(0.1)
			print("sim_state = ", sim_state)
			env.set_inner_state(sim_state)
			env.sim.forward()

			frame = env.env.sim.render(width=3000, height=1000, mode="offscreen")
			frames.append(frame)

			# viewer.scn.flags[2] = 0 # Disable reflections (~25% speedup)
			# body_id = env.sim.model.body_name2id(target)
			# lookat = env.sim.data.body_xpos[body_id]
			# for idx, value in enumerate(lookat):
			# 	viewer.cam.lookat[idx] = value
			# viewer.render()
			# img = viewer.read_pixels(1080, 1080, depth=False)
			# img = img[::-1, :, :]
			# frames.append(img)

		env.close()

		save_frames_as_video(frames, "/Users/chenu/Desktop/PhD/github/dcil_xpag/test/", train_it)

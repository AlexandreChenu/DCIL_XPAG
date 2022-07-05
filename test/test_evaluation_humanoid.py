#!/usr/bin python -w

import os

import gym
import gym_ghumanoid

import cv2
import pickle
import pdb

import time


def load_sim_traj(path):

	with open(path, 'rb') as handle:
		sim_traj = pickle.load(handle)

	return sim_traj

def save_frames_as_video(frames, path):

	video_name = path + '/video.mp4'
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

	traj_path = "/Users/chenu/Desktop/PhD/results_JZ/preliminary_results/preliminary_results_DCIL_humanoid/20220705_3920067/sim_traj_490000.pickle"
	sim_traj = load_sim_traj(traj_path)
	print(len(sim_traj))

	frames = []

	env.reset()

	for sim_state in sim_traj:
		# time.sleep(0.1)
		print("sim_state = ", sim_state)
		env.set_inner_state(sim_state)
		env.sim.forward()
		frame = env.env.sim.render(width=1080, height=1080, mode="offscreen")
		frames.append(frame)

	env.close()

	save_frames_as_video(frames, "/Users/chenu/Desktop/PhD/github/dcil_xpag/test/")

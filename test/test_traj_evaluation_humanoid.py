#!/usr/bin python -w

import os

import gym
import gym_ghumanoid

import cv2
import pickle
import pdb

import time

import mujoco_py

import matplotlib.pyplot as plt


def load_sim_traj(path):

	with open(path, 'rb') as handle:
		sim_traj = pickle.load(handle)

	return sim_traj

def load_demo_traj(path):
	with open(path, 'rb') as handle:
		demo_traj = pickle.load(handle)

	return demo_traj

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

	train_it = 940000
	# traj_path = "/Users/chenu/Desktop/PhD/results_JZ/preliminary_results/preliminary_results_DCIL_humanoid/v4/DCIL_v4_20220706_1957521/sim_traj_" + str(train_it) + ".pickle" #498000.pickle"
	traj_path = "/Users/chenu/Desktop/PhD/results_JZ/results_DCIL_TRO/DCIL_humanoid_long_demo/sim_traj_3288000.pickle"
	sim_traj = load_sim_traj(traj_path)
	print(len(sim_traj))

	## Extract body positions from evaluation traj
	# L_body = ["right_lower_arm", "left_lower_arm", "left_foot", "right_foot", "torso"]
	L_body = ["left_foot", "right_foot", "torso"]
	L_positions_eval = []
	env.reset()
	for sim_state in sim_traj:

		positions = {}

		# time.sleep(0.1)
		# print("sim_state = ", sim_state)
		env.set_inner_state(sim_state)
		env.sim.forward()

		for body in L_body:
			print("\n Body = ", body)
			body_id = env.sim.model.body_name2id(body)
			positions[body] = env.sim.data.body_xpos[body_id].copy()
			print("pos = ", positions[body])

		L_positions_eval.append(positions)


	## Extract body positions from demonstration
	demo_path = "/Users/chenu/Desktop/PhD/github/dcil_xpag/demos/humanoid_convert/1.demo" #498000.pickle"
	demo_traj = load_demo_traj(demo_path)

	# L_body = ["right_lower_arm", "left_lower_arm", "left_foot", "right_foot", "torso"]
	L_body = ["left_foot", "right_foot", "torso"]
	L_positions_demo = []
	env.reset()
	for sim_state in demo_traj["sim_states"][:500]:

		positions = {}

		# time.sleep(0.1)
		# print("sim_state = ", sim_state)
		env.set_inner_state(sim_state)
		env.sim.forward()

		for body in L_body:
			print("\n Body = ", body)
			body_id = env.sim.model.body_name2id(body)
			positions[body] = env.sim.data.body_xpos[body_id].copy()
			print("pos = ", positions[body])

		L_positions_demo.append(positions)

	env.close()

	print()

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	for body in L_body:
		X = [pos[body][0] for pos in L_positions_eval]
		Y = [pos[body][1] for pos in L_positions_eval]
		Z = [pos[body][2] for pos in L_positions_eval]
		print("X = ", X)

		ax.plot(X,Y,Z, label = body + "_eval")

	for body in L_body:
		X = [pos[body][0] for pos in L_positions_demo]
		Y = [pos[body][1] for pos in L_positions_demo]
		Z = [pos[body][2] for pos in L_positions_demo]
		print("X = ", X)

		ax.plot(X,Y,Z, label = body + "_demo")

	ax.set_ylim((-5,5.))
	plt.legend()
	plt.show()

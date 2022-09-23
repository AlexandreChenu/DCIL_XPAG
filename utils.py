import numpy as np 


def check_skill_matrix_valid(visited, success):
	'''
	The visitation count for a skill must be less than the visitation count for the last skill when overshooting.
	The SR of an overshoot must be less than the success of the preceding skill
	'''
	visited,success = np.array(visited), np.array(success)
	failed_overshoot = visited[..., :-1] - visited[..., 1:]
	visited_condition = failed_overshoot<0
	visited_condition = np.triu(visited_condition) # we don't care about the lower triangle matrix
	if visited_condition.any():
		print('problem with visited :\n',visited)
		raise Exception('invalid')

	sr_condition = success[..., :-1] < success[..., 1:]
	sr_condition = np.triu(sr_condition) 
	if sr_condition.any():
		print('problem with success :\n',success)
		raise Exception('invalid')
import numpy as np
import FP_Analysis as fp
import time
import matplotlib.pyplot as plt


def GetUnique(points, tol=1e-2):
	'''find all unique points in a set of noisy points'''
	#there must be at least one unique point
	unique_pts = [ [] ]		#list of lists (each list is a list of points corresponding to a unique root)
	clf_pts = []	#will hold indices of points that have already been grouped
	unclf_pts = []

    #first pass--point is classified identically to self
	for idx in range(len(points)):
		pt1 = points[0]
		pt2 = points[idx]
		curr_distance = fp.ComputeDistance(pt1, pt2)
		#if points are close enough together to be considered identical
		if curr_distance < tol:
			#store point idx in classified points
			clf_pts.append(idx)
			#if point not sufficiently far to be considered unique
			unique_pts[0].append(pt2)
		else:
			unclf_pts.append(idx)	#holds the indices of all points that are different from first point

	#keep classifying points untill all points have been classified
	while len(clf_pts) != len(points):
		unique_pts.append([])	#new sub-list of identical points
		unclf_pts_tmp = []		#reset to hold this round of unclassified pts
		pt1 = points[unclf_pts[0]]
		for idx in unclf_pts:
			pt2 = points[idx]
			curr_distance = fp.ComputeDistance(pt1, pt2)
			if curr_distance < tol:
				unique_pts[-1].append(pt2)	#start at zero so pt2=pt1 --> pt1 is included if it is truly unique to all other points
				clf_pts.append(idx)
			else:
				unclf_pts_tmp.append(idx)
		unclf_pts = unclf_pts_tmp	#reset unclassified points
	for _ in range(len(unique_pts)):
		unique_pts[_] = np.mean(unique_pts[_], axis=0)	#want to average over rows to get one point

	return unique_pts



def FindClosestTwo(points):
	'''will find closest two points from a set'''
	smallest_distance = 1e8		#assumes that there exists two points in the set with distance less than 1e8
	for pt1_idx in range(len(points)):
		for pt2_idx in range(pt1_idx+1, len(points)):
			curr_distance = fp.ComputeDistance(points[pt1_idx], points[pt2_idx])
			if curr_distance < smallest_distance:
				smallest_distance = curr_distance
				candidate_pt1 = points[pt1_idx]
				candidate_pt2 = points[pt2_idx]
	return candidate_pt1, candidate_pt2, smallest_distance


def GenerateLine(pt1, pt2):
	'''generates a line through 2 points in high dimensional space'''
	dimensionality = len(pt1)
	line = np.zeros((dimensionality, 100))
	for dim in range(dimensionality):
		line[dim,:] = np.linspace(pt1[dim], pt2[dim], 100)
	return line


def EvalPotential(F, pt, unit_vec):
	return np.matmul(F(pt), unit_vec)  


def PlotPotential(roots, F):
	#roots contains all the roots for a given input condition
	#we want to find how many unique values exit within roots

	#round each root in R50 to the nearest decimal
	#roots = np.around(roots, decimals=2)
	#print('\nrounded roots:\n', roots[:,:5])
	#unique_roots = np.unique(roots, axis=0)
	#print('\nunique_roots:\n', unique_roots[:,:].T)
	#print(':', len(unique_roots))


	unique_roots = roots#GetUnique(roots)
	#unique_roots is a list of arrays, where each array corresponds to a uique root

	#if there is only one root exit function
	if len(unique_roots) == 1:
		success=False
		return 0, success

	#if we have three roots find a line going through then earest two
	elif len(unique_roots) == 3:
		success=True
		#need to find the closer of two roots
		
		root1, root2, d = FindClosestTwo(unique_roots)
		if fp.IsAttractor(root1, F):
			difference_vector = root2-root1
		elif fp.IsAttractor(root2, F):
			difference_vector = root1-root2
		else: 
			return 0, False
		difference_mag = np.linalg.norm(difference_vector)
		unit_vector = (difference_vector) / difference_mag

		#give our line length 10
		extrapolation = ((40.0/difference_mag) - 1)/2.0
		#print('extrapolation coeff:', extrapolation)

		#now compute the line defined by these two points
		line = GenerateLine(root1-extrapolation*difference_vector, root2+extrapolation*difference_vector)

		#now we need to evalue F dot unit vector at every point along the line
		potentials = np.zeros((100))
		for _ in range(len(line[0,:])):			#loop through columns corresponding to points along line (rows correspond to dimensions)
			pt = line[:,_]						#grab current high dimensional point off the line
			pt_potential = EvalPotential(F, pt, unit_vector)	#evaluate the poential at this point along the line
			potentials[_] = pt_potential

		fig = plt.figure(76)		#inspired by Fallout 76
		ax = fig.add_subplot(1, 1, 1)

		# set the x-spine
		ax.spines['left'].set_position('zero')

		# turn off the right spine/ticks
		ax.spines['right'].set_color('none')
		ax.yaxis.tick_left()

		# set the y-spine
		ax.spines['bottom'].set_position('zero')

		# turn off the top spine/ticks
		ax.spines['top'].set_color('none')
		ax.xaxis.tick_bottom()

		plt.plot(np.linspace(0,100,100), potentials)
		return potentials, success
	else:
		print('Potential calculation for other than 1 or 3 roots is not yet supported!')
		return 0, False
# -*- coding: utf-8 -*-
#
# This file is part of PyGaze - the open-source toolbox for eye tracking
#
#	PyGazeAnalyser is a Python module for easily analysing eye-tracking data
#	Copyright (C) 2014  Edwin S. Dalmaijer
#
#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <http://www.gnu.org/licenses/>

# EyeTribe Reader
#
# Reads files as produced by PyTribe (https://github.com/esdalmaijer/PyTribe),
# and performs a very crude fixation and blink detection: every sample that
# is invalid (usually coded '0.0') is considered to be part of a blink, and
# every sample in which the gaze movement velocity is below a threshold is
# considered to be part of a fixation. For optimal event detection, it would be
# better to use a different algorithm, e.g.:
# Nystrom, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation,
# saccade, and glissade detection in eyetracking data. Behavior Research
# Methods, 42, 188-204. doi:10.3758/BRM.42.1.188
#
# (C) Edwin Dalmaijer, 2014
# edwin.dalmaijer@psy.ox.ax.uk
#
# version 1 (01-Jul-2014)

__author__ = "Edwin Dalmaijer"
import matplotlib.pyplot as plt

import numpy


def blink_detection(x, y, time, missing=0.0, minlen=10):
	
	"""Detects blinks, defined as a period of missing data that lasts for at
	least a minimal amount of samples
	
	arguments

	x		-	numpy array of x positions
	y		-	numpy array of y positions
	time		-	numpy array of EyeTribe timestamps

	keyword arguments

	missing	-	value to be used for missing data (default = 0.0)
	minlen	-	integer indicating the minimal amount of consecutive
				missing samples
	
	returns
	Sblk, Eblk
				Sblk	-	list of lists, each containing [starttime]
				Eblk	-	list of lists, each containing [starttime, endtime, duration]
	"""
	
	# empty list to contain data
	Sblk = []
	Eblk = []
	
	# check where the missing samples are
	mx = numpy.array(x==missing, dtype=int)
	my = numpy.array(y==missing, dtype=int)
	miss = numpy.array((mx+my) == 2, dtype=int)
	
	# check where the starts and ends are (+1 to counteract shift to left)
	diff = numpy.diff(miss)
	starts = numpy.where(diff==1)[0] + 1
	ends = numpy.where(diff==-1)[0] + 1
	
	# compile blink starts and ends
	for i in range(len(starts)):
		# get starting index
		s = starts[i]
		# get ending index
		if i < len(ends):
			e = ends[i]
		elif len(ends) > 0:
			e = ends[-1]
		else:
			e = -1
		# append only if the duration in samples is equal to or greater than
		# the minimal duration
		if e-s >= minlen:
			# add starting time
			Sblk.append([time[s]])
			# add ending time
			Eblk.append([time[s],time[e],time[e]-time[s]])
	
	return Sblk, Eblk

def remove_missing(x, y, time, missing):
	mx = numpy.array(x==missing, dtype=int)
	my = numpy.array(y==missing, dtype=int)
	x = x[(mx+my) != 2]
	y = y[(mx+my) != 2]
	time = time[(mx+my) != 2]
	return x, y, time


def fixation_detection(x, y, time, missing=0.0, maxdist=25, mindur=100):
	
	"""Detects fixations, defined as consecutive samples with an inter-sample
	distance of less than a set amount of pixels (disregarding missing data)
	
	arguments

	x		-	numpy array of x positions
	y		-	numpy array of y positions
	time		-	numpy array of EyeTribe timestamps

	keyword arguments

	missing	-	value to be used for missing data (default = 0.0)
	maxdist	-	maximal inter sample distance in pixels (default = 25)
	mindur	-	minimal duration of a fixation in milliseconds; detected
				fixation cadidates will be disregarded if they are below
				this duration (default = 100)
	
	returns
	Sfix, Efix
				Sfix	-	list of lists, each containing [starttime]
				Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
	"""

	x, y, time = remove_missing(x, y, time, missing)


	# empty list to contain data
	Sfix = []
	Efix = []
	
	# loop through all coordinates
	si = 0
	fixstart = False
	for i in range(1,len(x)):
		# calculate Euclidean distance from the current fixation coordinate
		# to the next coordinate
		squared_distance = ((x[si]-x[i])**2 + (y[si]-y[i])**2)
		dist = 0.0
		if squared_distance > 0:
			dist = squared_distance**0.5
		# check if the next coordinate is below maximal distance
		if dist <= maxdist and not fixstart:
			# start a new fixation
			si = 0 + i
			fixstart = True
			Sfix.append([time[i]])
		elif dist > maxdist and fixstart:
			# end the current fixation
			fixstart = False
			# only store the fixation if the duration is ok
			if time[i-1]-Sfix[-1][0] >= mindur:
				Efix.append([Sfix[-1][0], time[i-1], time[i-1]-Sfix[-1][0], x[si], y[si]])
			# delete the last fixation start if it was too short
			else:
				Sfix.pop(-1)
			si = 0 + i
		elif not fixstart:
			si += 1
	#add last fixation end (we can lose it if dist > maxdist is false for the last point)
	if len(Sfix) > len(Efix):
		Efix.append([Sfix[-1][0], time[len(x)-1], time[len(x)-1]-Sfix[-1][0], x[si], y[si]])
	return Sfix, Efix


def saccade_detection(x, y, time, missing=0.0, minlen=40, maxvel=1, maxacc=0):
	
	"""Detects saccades, defined as consecutive samples with an inter-sample
	velocity of over a velocity threshold or an acceleration threshold
	
	arguments

	x		-	numpy array of x positions
	y		-	numpy array of y positions
	time		-	numpy array of tracker timestamps in milliseconds

	keyword arguments

	missing	-	value to be used for missing data (default = 0.0)
	minlen	-	minimal length of saccades in milliseconds; all detected
				saccades with len(sac) < minlen will be ignored
				(default = 5)
	maxvel	-	velocity threshold in pixels/second (default = 40)
	maxacc	-	acceleration threshold in pixels / second**2
				(default = 340)
	
	returns
	Ssac, Esac
			Ssac	-	list of lists, each containing [starttime]
			Esac	-	list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
	"""
	x, y, time = remove_missing(x, y, time, missing)

	# CONTAINERS
	Ssac = []
	Esac = []
	

	dx = numpy.diff(x)
	dy = numpy.diff(y)
	dt = numpy.diff(time)
	vel = numpy.sqrt(dx**2 + dy**2) / dt
    # Find indices where velocity exceeds threshold
	# saccade_indices = numpy.where(velocities > 1)[0]

    # Calculate number of saccades
	# num_saccades = len(saccade_indices)
	# print(num_saccades)
	# INTER-SAMPLE MEASURES
	# the distance between samples is the square root of the sum
	# of the squared horizontal and vertical interdistances
	# intdist = (numpy.diff(x)**2 + numpy.diff(y)**2)**0.5
	# # get inter-sample times
	# inttime = numpy.diff(time)
	
	# recalculate inter-sample times to seconds
	# inttime = inttime / 1000.0
	
	# VELOCITY AND ACCELERATION
	# the velocity between samples is the inter-sample distance
	# divided by the inter-sample time
	# vel = intdist / inttime
	# the acceleration is the sample-to-sample difference in
	# eye movement velocity
	acc = numpy.diff(vel)

	# SACCADE START AND END
	t0i = 0
	stop = False
	while not stop:
		# saccade start (t1) is when the velocity or acceleration
		# surpass threshold, saccade end (t2) is when both return
		# under threshold
	
		# detect saccade starts
		sacstarts = numpy.where((vel[1+t0i:] > maxvel).astype(int) + (acc[t0i:] > abs(maxacc)).astype(int) >= 1)[0]
		if len(sacstarts) > 0:
			# timestamp for starting position
			t1i = t0i + sacstarts[0] + 1
			if t1i >= len(time)-1:
				t1i = len(time)-2
			t1 = time[t1i]
			
			# add to saccade starts
			Ssac.append([t1])
			
			# detect saccade endings
			sacends = numpy.where((vel[1+t1i:] < maxvel).astype(int) + (acc[t1i:] < maxacc).astype(int) == 2)[0]
			if len(sacends) > 0:
				# timestamp for ending position
				t2i = sacends[0] + 1 + t1i + 2
				if t2i >= len(time):
					t2i = len(time)-1
				t2 = time[t2i]
				dur = t2 - t1

				# ignore saccades that did not last long enough
				# print(dur,minlen)
				if dur >= minlen:
					# add to saccade ends
					Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
				else:
					# remove last saccade start on too low duration
					Ssac.pop(-1)

				# update t0i
				t0i = 0 + t2i
			else:
				stop = True
		else:
			stop = True


	return Ssac, Esac


import numpy as np

def calculate_saccades(x, y, time, velocity_threshold = 2,missing= 0.0):

	x, y, time = remove_missing(x, y, time, missing)
    # Calculate velocities using the change in position over time
	dx = np.diff(x)
	dy = np.diff(y)
	dt = np.diff(time)
	velocities = np.sqrt(dx**2 + dy**2) / dt
    # Find indices where velocity exceeds threshold
	saccade_indices = np.where(velocities > velocity_threshold)[0]

    # Initialize list to store saccade information
	saccades = []
	Ssac = []
	Esac = []
    # Iterate through saccade indices to calculate properties
	for index in saccade_indices:
		start_index = max(0, index - 1)
		end_index = min(len(x) - 1, index + 1)
		start_time = time[start_index]
		end_time = time[end_index]
		duration = end_time - start_time
		start_x = x[start_index]
		start_y = y[start_index]
		end_x = x[end_index]
		end_y = y[end_index]
		Ssac.append(start_time)
		Esac.append([start_time, end_time, duration, start_x, start_y, end_x, end_y])
		saccades.append({
			"start_time": start_time,
			"end_time": end_time,
			"duration": duration,
			"start_x": start_x,
			"start_y": start_y,
			"end_x": end_x,
			"end_y": end_y
		})

	# plt.ylim(210,0)
	# plt.xlim(0,160)
	# plt.scatter(x,y)
	# plt.show()	
	# print(Ssac,Esac)
	return Ssac, Esac

# # Example usage:
# x = [0, 1, 3, 4, 5, 7, 8, 9, 10]  # X coordinates
# y = [0, 2, 3, 4, 5, 6, 7, 8, 9]     # Y coordinates
# time = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Time samples (arbitrary units)
# velocity_threshold = 1.0           # Adjust as needed

# saccades = calculate_saccades(x, y, time, velocity_threshold)

# # Print saccade information
# for i, saccade in enumerate(saccades):
#     print(f"Saccade {i+1}:")
#     print("Start Time:", saccade["start_time"])
#     print("End Time:", saccade["end_time"])
#     print("Duration:", saccade["duration"])
#     print("Start X:", saccade["start_x"])
#     print("Start Y:", saccade["start_y"])
#     print("End X:", saccade["end_x"])
#     print("End Y:", saccade["end_y"])
#     print()
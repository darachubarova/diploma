import pandas as pd
import numpy as np
from time import time

def functional1_old(paterns):
    v1 = np.repeat(paterns, len(paterns), axis=0)
    v2 = np.repeat(np.array([paterns]), len(paterns),
                    axis=0).reshape(len(paterns) ** 2, len(paterns[-1]))
    return np.sum((v1 - v2) ** 2 )

def functional1(patterns):
    dist_sum = 0
    for i in range (len(patterns)):
        for j in range (i+1, len(patterns)):
            dist_sum += np.sum((patterns[i] - patterns[j])**2)
    return dist_sum / (2 * len(patterns) ** 2)

def functional2(vectors):
    mean_vec = vectors.mean(axis=0)
    return np.sum((vectors - mean_vec) ** 2) / len(vectors)

def check_pattern(v0, v, v_d):
    return (np.abs(v0 - v) <= v_d).all(axis=1) * (np.abs(v0[:, 1:]  - v[:, 1:]) <= v_d[:, :-1]).all(axis=1)

def find_patterns(file, eps = 0.01, h=0, tube_type='fixed', type_of_clasterisation="abs", try_times=100):

    if type_of_clasterisation == "tan":
        data = file[:, 1:] - file[:, :-1]
    else:
        data = np.copy(file)

    num_vectors = np.arange(data.shape[0])

    patterns = list()
    patterns_delta = list()
    num_patterns = list()

    for _ in range(try_times):
        patterns.append(list())
        num_patterns.append(np.zeros((file.shape[0], 1), dtype=int))
        mask = np.array([True] * data.shape[0]) #те, кто еще не нашел себе патерн

        pattern_num = 0

        distance_in_pattern = 0

        while mask.any():

            index = np.random.randint(num_vectors[mask].shape[0])
            num_base_vector = num_vectors[mask][index] #number of the base vector
            base_vector = data[num_base_vector]
            base_vectors = np.ones(file.shape[0]).reshape((-1, 1)) @ base_vector.reshape((1, -1))

            if tube_type == 'fixed':
                d = np.ones(data.shape) * eps
            elif tube_type == 'adaptive':
                d = np.abs(base_vectors) * eps + h
            else:
                d = (np.max(np.abs(base_vectors), axis=1)).reshape((-1, 1)) @ np.ones(base_vectors.shape[1]).reshape((1, -1)) * eps + h

            new_mask = check_pattern(base_vectors, data, d) #эти попали в трубку

            pattern_mask = new_mask * mask
            num_patterns[-1][pattern_mask] = pattern_num
            patterns[-1].append(file[pattern_mask].mean(axis=0))

            distance_in_pattern += functional2(file[pattern_mask])

            mask *= ~new_mask
            pattern_num += 1


        dp1 = functional1(patterns[-1])
        dp2 = distance_in_pattern / len(patterns[-1])
        patterns_delta.append(dp1 - dp2)

    best_try = np.argmax(patterns_delta)
    return num_patterns[best_try]


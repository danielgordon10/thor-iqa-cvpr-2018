import numpy as np
import json
import glob
import utils
from utils import game_util

from constants import AGENT_STEP_SIZE

files = sorted(glob.glob('layouts/*.json'), key=lambda x: int(x.split('FloorPlan')[1].split('-')[0]))
start_positions = np.load('layouts/start_positions.npy')
numPoints = []
minX = 10
minY = 10
maxX = -10
maxY = -10

maxRangeX = -1
maxRangeY = -1

files = files[:30]

for ff,file_name in enumerate(files):
    print('')
    print(file_name)
    f = json.load(open(file_name))
    points = [[point['x'], point['z']]
        for point in f['allPoints'] if len(point['receptacleObjectId']) == 0]
    points.append(start_positions[ff,:])
    if len(points) < 20:
        print('###### Error very few points for scene ', file_name)
    points = np.array(points)
    rangeX = np.max(points[:,0]) - np.min(points[:,0])
    rangeY = np.max(points[:,1]) - np.min(points[:,1])
    maxRangeX = max(rangeX, maxRangeX)
    maxRangeY = max(rangeY, maxRangeY)
    print('max range', maxRangeX / AGENT_STEP_SIZE, maxRangeY / AGENT_STEP_SIZE)
    minX = min(9999, np.min(points[:,0]))
    minY = min(9999, np.min(points[:,1]))
    maxX = max(-999, np.max(points[:,0]))
    maxY = max(-999, np.max(points[:,1]))
    print('room min and max', minX, minY, maxX, maxY)
    points = game_util.unique_rows(points)
    numPoints.append(len(points))
    points = (points * 1.0 / AGENT_STEP_SIZE).astype(int)
    points = game_util.unique_rows(points)
    points = points.astype(float) * AGENT_STEP_SIZE
    np.save(file_name[:-4] + 'npy', points)






#!/usr/bin/env python

#*************************************************
#    Filename: station_split.py
#    Author: msy - shumao@wethz.ch
#    Description: ---
#    Create: 2023-07-06 22:22:48
#    Last Modified: 2023-07-06 22:22:48
#*************************************************
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from itertools import chain

def divide_map_into_grid(x0, y0, map_width, map_height, grid_width, grid_height):
    num_columns = map_width // grid_width
    num_rows = map_height // grid_height
    
    grid = [[None for _ in range(num_columns)] for _ in range(num_rows)]
    
    for row in range(num_rows):
        for col in range(num_columns):
            x = x0 + col*grid_width
            y = y0 + row*grid_height
            grid[row][col] = (x, y, x+grid_width, y+grid_height)
    return grid

def count_stations_in_grid(grid, stations):
    station_counts = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]

    nsta = stations['name'].values.shape[0]
    nrow = len(grid)
    ncol = len(grid[0])
    grid_num = np.zeros(nsta)
    for i in range(nsta):
        lat = stations['lat'][i]
        lon = stations['lon'][i]
        for row_idx, row in enumerate(grid):
            for col_idx, cell in enumerate(row):
                cell_x1, cell_y1, cell_x2, cell_y2 = cell
                if cell_x1 <= lon <= cell_x2 and cell_y1 <= lat <= cell_y2:
                    # print('grid: ', row_idx, col_idx)
                    station_counts[row_idx][col_idx] += 1
                    grid_num[i] = row_idx*ncol + col_idx
    stations['grid'] = grid_num
    return station_counts
            
map_width = 360
map_height = 180
grid_width = 120
grid_height = 30

grid = divide_map_into_grid(-180, -90, map_width, map_height, grid_width, grid_height)
nrow = len(grid)
ncol = len(grid[0])
x1, y1, x2, y2 = zip(*chain.from_iterable(grid))
print(nrow, ncol)

fsit = sys.argv[1]
stas = pd.read_csv(fsit, sep='\s+', names=['name', 'lat', 'lon', 'height'])

station_counts = count_stations_in_grid(grid, stas)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.stock_img()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])
 
# msize = 50
# ax.scatter(x1, y1, color='b', s=msize, marker='+')
# ax.scatter(x2, y2, color='b', s=msize, marker='+')
# for row_idx, row in enumerate(grid):
#     for col_idx, cell in enumerate(row):
#         mask = (stas['grid'] == row_idx*ncol + col_idx)
#         ax.scatter(stas[mask]['lon'], stas[mask]['lat'], label=f'({row_idx}, {col_idx})')
# ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=10)

nsta = len(stas)
train_per = 0.8
test_per = 0.2
train_stations = []
test_stations = []
itrain = 0
itest = 0
for row_idx, row in enumerate(grid):
    for col_idx, cell in enumerate(row):
        n = station_counts[row_idx][col_idx]    
        if n == 0:
            continue
        mask = (stas['grid'] == row_idx*ncol+col_idx)
        stas_shuffle = stas[mask].sample(frac=1, random_state=72)
        ntrain = round(n*0.8)
        ntest = n - ntrain
        if ntrain == 0:
            continue
        for i in range(ntrain):
            train_stations.append(stas_shuffle.iloc[i])
        for i in range(ntest):
            test_stations.append(stas_shuffle.iloc[ntrain+i])
train_stations = pd.DataFrame(train_stations)
train_stations = train_stations.reset_index(drop=True)
test_stations = pd.DataFrame(test_stations)
test_stations = test_stations.reset_index(drop=True)
train_stations.to_csv('train.csv', index=None, float_format='%10.2f')
test_stations.to_csv('test.csv', index=None, float_format='%10.2f')
train_list = np.sort(np.array(train_stations['name'].values))
test_list = np.sort(np.array(test_stations['name'].values))
np.savetxt('train.list', train_list, fmt='%s')
np.savetxt('test.list', test_list, fmt='%s')

msize = 25
ax.scatter(train_stations['lon'], train_stations['lat'], s=msize, 
           color='b', linewidth=0.1, edgecolor='k', label='Train', zorder=10)
ax.scatter(test_stations['lon'], test_stations['lat'], s=msize, 
           color='r', linewidth=0.1, edgecolor='k', label='Test', zorder=11)
ax.legend(loc='lower left', bbox_to_anchor=(0.005, 0.001), 
          handletextpad=0.1, borderpad=0.1)
plt.savefig('output.png', bbox_inches='tight')

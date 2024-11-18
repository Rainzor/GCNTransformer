# Install required packages.
import os
# Helper function for visualization.
import matplotlib.pyplot as plt
import numpy as np


# datafile = './data/PlateauBorder/raw128/stdata.bin'
datafile = "D:/Downloads/foamGNN/raw/2/stdata.bin"

stdata = np.fromfile(datafile, dtype=np.float32)
print(stdata.shape)
print(np.sum(stdata>0))
sizefile = './data/PlateauBorder/rawz/stnum.txt'
sizes = np.loadtxt(sizefile, dtype=np.int32)
# print(sizes)

stdata = stdata.reshape(sizes)

sum = np.sum(stdata)
print(sum)

print(stdata.shape)

stdata_pos = np.sum(stdata, axis=(2, 3))
stdata_pos = stdata_pos / np.sum(stdata_pos) * sizes[0]*sizes[1] / np.pi/4
print("Shape of positive data: ", stdata_pos.shape)
print("Sum of positive data: ", np.sum(stdata_pos))

stdata_dir = np.sum(stdata, axis=(0, 1))
stdata_dir = stdata_dir / np.sum(stdata_dir) * sizes[2]*sizes[3] / np.pi/4
print("Shape of directional data: ", stdata_dir.shape)
print("Sum of directional data: ", np.sum(stdata_dir))

stdata = stdata / np.sum(stdata) * np.prod(sizes) / np.pi/np.pi/16


# 定义 z 和 theta 的取值范围
z_in = np.linspace(-1, 1, 32)  # 从南极到北极，高度范围 [-1, 1]
theta_in = np.linspace(-np.pi, np.pi, 64)  # 球面上横向角度 [-π, π]

# 生成网格
Z, THETA = np.meshgrid(z_in, theta_in, indexing='ij')


# 将 z 和 theta 转换为球面坐标 (r, theta, phi)
# 假设球体的半径为 1
R = np.sqrt(1 - Z**2)
PHI = THETA

# 将球面坐标转换为笛卡尔坐标
X = R * np.cos(PHI)
Y = R * np.sin(PHI)

# 绘图
fig = plt.figure('拟合图')
ax = fig.add_subplot(projection='3d')

# 在球面上绘制数据
ax.plot_surface(X, Y, Z, facecolors=plt.cm.rainbow(stdata_dir), rstride=1, cstride=1)


plt.show()
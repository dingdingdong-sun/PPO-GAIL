from job import Workloads
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
SEQUENCE_LENGTH = 512

load = Workloads("./data/HPC2N-2002-2.2-cln.swf")

requests_resource = []
job_size = []
requests_resource_filter = []
job_size_after_filter = []
count = 0
fig = plt.figure()

def plot_job_distribution(requests_resource, job_size, sub,t):
    requests_resource = np.array(requests_resource, dtype=np.int32)
    job_size = np.array(job_size, dtype=np.int32)

    max_requests_resource = np.max(requests_resource)
    x_steps = np.floor_divide(np.log10(max_requests_resource), 0.25) * 0.25
    # min_requests_resource = np.min(requests_resource)
    # step_requests_resource = (max_requests_resource - min_requests_resource)/10.0
    z1 = np.floor_divide(np.log10(requests_resource), 0.25) * 0.25
    max_job_size = np.max(job_size)
    y_steps = np.floor(np.log10(max_job_size))
    # min_job_size = np.min(job_size)
    # step_job_size = (max_job_size - min_job_size)/10.0
    z2 = np.floor_divide(np.log10(job_size), 1)

    Z = np.zeros(shape=(int(x_steps * 4), int(y_steps)))
    X = np.arange(0, x_steps, step=0.25)
    Y = np.arange(0, y_steps, step=1)

    l = len(z1)
    for i in range(l):
        if z1[i] == x_steps:
            z1[i] -= 0.25
        if int(z2[i]) == y_steps:
            z2[i] -= 1
        Z[int(z1[i] * 4), int(z2[i])] += 1

    xx, yy = np.meshgrid(X, Y)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
    bottom = np.zeros_like(X)  # 设置柱状图的底端位值
    Z = Z.ravel()  # 扁平化矩阵

    # 绘图设置

    ax = fig.add_subplot(1, 2, sub, projection='3d')  # 三维坐标轴
    ax.bar3d(X, Y, bottom, 0.25, 1, Z, shade=True)  #
    # 坐标轴设置
    ax.set_title(t)
    ax.set_xlabel('requests_resource(10^x)')
    ax.set_ylabel('job_size(10^x)')
    ax.set_zlabel('number')

# for j in load.all_jobs:
#     if count%50000 == 0:
#         requests_resource.append([])
#         job_size.append([])
#     requests_resource[count//50000].append(j.request_number_of_processors)
#     job_size[count//50000].append(j.run_time)
#     count += 1
for j in load.all_jobs:
    requests_resource.append(j.request_number_of_processors)
    job_size.append(j.run_time)
    count += 1

with open("first_filter_index.txt", "r") as f:
    tmp = f.readline().strip().split(",")
    start_index = [int(x) for x in tmp]
# for j in load.all_jobs:
#
#     requests_resource.append(j.request_number_of_processors)
#     job_size.append(j.run_time)
#     # count += 1
#
# requests_resource = np.array(requests_resource)
# job_size = np.array(job_size)
# print("max runtime:{}, min_runtime:{}, max resource:{}, min_resource:{}".format(np.max(job_size), np.min(job_size), np.max(requests_resource), np.min(requests_resource)))

i = 0
i_max = len(start_index)-1

l = len(job_size)
for j in range(len(load.all_jobs)):
    if j < start_index[i]:
        continue
    elif j-start_index[i]<512:
        requests_resource_filter.append(load[j].request_number_of_processors)
        job_size_after_filter.append(load[j].run_time)
    else:
        i += 1
        if i > i_max:
            break
# for i in range(4):
#     plot_job_distribution(requests_resource[i], job_size[i], i+1)

plot_job_distribution(requests_resource,job_size,1, "HPC2n before filter")
plot_job_distribution(requests_resource_filter,job_size_after_filter,2, "HPC2N after filter")
plt.show()

# users_competition = []
# resource_competition = []



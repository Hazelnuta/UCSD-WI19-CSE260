from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np


def load(filename):
    d = [x.strip().split() for x in open(filename, 'r').readlines()]
    x = [int(i[0])for i in d]
    y = [float(i[1])for i in d]
    return x, y


PEAK_GFLOPS = 1870
MEM_BW = 240
sns.set_style('whitegrid')
plt.figure(figsize=(12, 6))
# plt.xlabel('Size of Matrix (N)')
# plt.xlabel('Flop / Double Word')
plt.ylabel('Gflops')
# plt.ylabel('Speed-up Ratio (GPU / CPU)')
# plt.xticks([0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
plt.xticks([32, 256, 512, 768, 1024, 2048])
# plt.xticks([32, 64, 128, 256])
# x = np.arange(0.1, 512.1, 0.1)
# y = [min(PEAK_GFLOPS, 240 / 8 * i) for i in x]
# x240 = x[np.searchsorted(y, 844)]
# s = sns.lineplot(x=x, y=y, label='memory bandwidth = 240 GB/s')
# x = list(np.arange(0.1, 512.1, 0.1))
# y = [min(PEAK_GFLOPS, 154 / 8 * i) for i in x]
# x154 = x[np.searchsorted(y, 844)]
# y154 = y[x.index(x240)]
# s = sns.lineplot(x=x, y=y, label='memory bandwidth = 154 GB/s')
# s.axes.set_xscale('log')
# s.axes.set_yscale('log')
# s.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
#               64.0, 128.0, 256.0, 512.0] + [x240, x154])
# s.set_yticks([16, 32, 64, 128, 256,  844, 1024, 2048] + [y154])
# plt.axhline(y=844, c='black', linestyle='dashed')
# plt.axhline(y=y154, c='black', linestyle='dashed')
# plt.axvline(x=x154, c='black', linestyle='dashed')
# plt.axvline(x=x240, c='black', linestyle='dashed')
# s.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# s.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


maxy = 0


def plot(filename, label=None):
    global maxy
    x, y = load('results/%s' % filename)
    maxy = max(maxy, max(y))
    label = label if label else '.'.join(filename.split('.')[:-1])
    sns.lineplot(x=x, y=y, label=label)


# cx, cy = load('results/mixed-cutlass.result')
# x = [256, 512, 768, 1023, 1024, 1025, 2047, 2048, 2049]
# y = [5.84, 17.4, 45.3, 73.7, 73.6, 73.5, 171, 182, 175]
# for i, n in enumerate(x):
#     y[i] = cy[cx.index(n)] / y[i]
# sns.lineplot(x=x, y=y, label='GPU / CPU Gflops ratio')
# sns.lineplot(x=x, y=y, label='BLAS')

# plot('64-64-16-8-4-cutlass.result')
plot('64-64-8-8-4-cutlass.result', label='BN=64 BM=64 BK=8 TN=8 TM=4')
# plot('64-64-4-8-4-cutlass.result')

plot('32-32-4-4-2-cutlass.result', label='BN=32 BM=32 BK=4 TN=4 TM=4')
plot('mixed-cutlass.result', label='mixed block')

# plot('64-64-16-8-4-cutlass.result')
# plot('64-64-16-4-4-cutlass.result')
# plot('64-64-16-8-4-cutlass.result')


# plot('128-64-32-8-4-cutlass.result')
# plot('64-64-16-8-4-cutlass.result')
# plot('64-64-16-4-4-cutlass.result')
# plot('64-64-32-8-4-cutlass.result')
# plot('64-64-8-8-4-cutlass.result')
# plot('64-64-4-8-4-cutlass.result')
# plot('naive32.result', label='Naive BX=BY=32')
# plot('naive16.result', label='Naive BX=BY=16')
# plot('mixed-cutlass.result')

# plt.yticks([100, 200, 300, 360, 400, 450, 500, 600, 700, 800, int(maxy)])
plt.yticks(list(range(100, int(maxy), 100)) + [int(maxy)])


# plt.savefig('test.png')
x, y = load('results/mixed-cutlass.result')
p = sorted(list(zip(x, y)), key=lambda s: -float(s[1]))
print(p[:20])

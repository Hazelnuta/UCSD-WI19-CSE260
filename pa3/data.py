import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import code

files = ['pa3-fused-novec.out', 'pa3-fused-vec.out',
         'pa3-unfused-novec.out', 'pa3-unfused-vec.out',
         'pa3-24.out', 'pa3-48.out', 'pa3-96.out', 'pa3-192.out',
         'pa3-384.out', 'pa3-480.out', 'pa3-960.out', 'pa3-ref.out']
data = []
for f in files:
    l = open('results/' + f, 'r').readlines()
    for x in l:
        [t, N, M, x, y, comm, i, Tp, gflops, Linf, L2] = x.split()
        d = {}
        d['type'] = t
        d['N'] = int(N)
        d['x'] = int(x)
        d['y'] = int(y)
        d['comm'] = comm
        d['iter'] = int(i)
        d['Tp'] = float(Tp)
        d['gflops'] = float(gflops)
        d['Linf'] = float(Linf)
        d['L2'] = float(L2)
        d['cores'] = int(x) * int(y)
        data.append(d)
df = pd.DataFrame(data)
# for n in [24, 48, 96, 192, 384, 480]:
#     mx = df[(df['cores'] == n) & (df['comm'] == 'Y')]['gflops'].max()
#     print(df[(df['cores'] == n) & (df['comm'] == 'Y')
#              & (df['gflops'] == mx)].iloc[0])
# print(df[df['comm'] == 'Y'].groupby(['cores', 'comm', 'type'])['gflops'].max())
# for n in [24, 48, 96, 192, 384, 480, 960]:
#     print(n)
#     for k in ['Y', 'N']:
#         for t in ['fused-novec', 'fused-vec', 'unfused-novec', 'unfused-vec', 'ref']:
#             print(int(df[(df['cores'] == n) & (df['type'] == t) & (df['comm'] == k)]
#                       ['gflops'].max() + 0.5), end=' & ')
#     print()
# for n in [1, 2, 4, 8, 24, 48, 96, 192, 384, 480, 960]:
#     print(n)
#     for t in ['fused-novec', 'fused-vec', 'unfused-novec', 'unfused-vec', 'ref']:
#         g1 = df[(df['cores'] == n) & (df['type'] == t)
#                 & (df['comm'] == 'Y')]['gflops'].max()
#         g2 = df[(df['cores'] == n) & (df['type'] == t)
#                 & (df['comm'] == 'N')]['gflops'].max()
#
#         o = (g2 - g1) / g1 * 100
#         print('%.2f\\%%' % o, end=' & ')
#     print()
#
for n in [1, 2, 4, 8, 24, 48, 96, 192, 384, 480, 960]:
    tmp = df[(df['cores'] == n) & (df['comm'] == 'Y')]['gflops'].max()
    tmp = df[(df['cores'] == n) & (df['comm'] == 'Y')
             & (df['gflops'] == tmp)].iloc[0]
    print(tmp['cores'], tmp['N'], tmp['x'], tmp['y'], tmp['iter'],
          tmp['Tp'], tmp['gflops'], '%E' % tmp['Linf'], '%E' % tmp['L2'], sep=', ')
# for n in [1, 2, 4, 8, 24, 48, 96, 192, 384, 480, 960]:
#     for r in df[(df['cores'] == n) & (df['comm'] == 'Y')][['cores', 'N', 'x', 'y', 'iter', 'Tp', 'gflops', 'Linf', 'L2']].values:
#         print(int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), r[5], r[6], '%E' %
#               r[7], '%E' % r[8], sep=', ')

# code.interact(local=locals())
# sns.set_style('whitegrid')
# plt.figure(figsize=(12, 6))
# plt.ylabel('Gflops')
# plt.xlabel('# of processors')
# xs = [1, 2, 4, 8, 24, 48, 96, 192, 384, 480, 960]
# plt.xticks(xs)
# for t in ['fused-novec', 'fused-vec', 'unfused-novec', 'unfused-vec', 'ref']:
#     ys = []
#     for x in xs:
#         g = df[(df['type'] == t) & (df['cores'] == x)
#                & (df['comm'] == 'Y')]['gflops'].max()
#         ys.append(g)
#     s = sns.lineplot(x=xs, y=ys, label=t)
#     s.axes.set_xscale('log')
#     s.set_xticks(xs)
#     s.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#
# plt.savefig('cores2gflops.png')


# plt.ylabel('Overhead')
# plt.xlabel('# of processors')
# for t in ['fused-novec', 'fused-vec', 'unfused-novec', 'unfused-vec', 'ref']:
#     ys = []
#     for n in [1, 2, 4, 8, 24, 48, 96, 192, 384, 480, 960]:
#         g1 = df[(df['cores'] == n) & (df['type'] == t)
#                 & (df['comm'] == 'Y')]['gflops'].max()
#         g2 = df[(df['cores'] == n) & (df['type'] == t)
#                 & (df['comm'] == 'N')]['gflops'].max()
#
#         o = (g2 - g1) / g1 * 100
#         ys.append(o)
#     s = sns.lineplot(x=xs, y=ys, label=t.replace('vec', 'sse'))
#     s.axes.set_xscale('log')
#     s.set_xticks(xs)
#     s.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.savefig('cores2overhead.png')

# plt.ylabel('Gflops')
# plt.xlabel('x/y')
# for t in ['fused-novec', 'fused-vec', 'unfused-novec', 'unfused-vec', 'ref']:
#     xs, ys = [], []
#     for r in df[(df['cores'] == 480) & (df['comm'] == 'Y')
#                 & (df['type'] == t)][['x', 'y', 'gflops']].values:
#         xs.append(r[0] / r[1])
#         ys.append(r[2])
#     s = sns.lineplot(x=xs, y=ys, label=t.replace('vec', 'sse'))
#     s.axes.set_xscale('log')
#     s.set_xticks(xs)
# s.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.savefig('480geo.png')

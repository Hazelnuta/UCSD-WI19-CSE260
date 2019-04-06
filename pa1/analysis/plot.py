from matplotlib import pyplot as plt
import seaborn as sns
import os
import re

BLAS_FNAME = '../results/best/blas-best.result'
BLOCK_FNAME = '../results/best/blocked-best.result'
GOTO_FNAME = '../results/best/goto-best.result'
NAIVE_FNAME = '../results/best/naive-best.result'
NAIVET_FNAME = '../results/best/naive-transpose-best.result'

GOTOL1 = '../results/goto-l1'
GOTOL2 = '../results/goto-l2'
GOTOL3 = '../results/goto-l3'

AVX = '../results/large'


def extract(fname):
    xs, ys = [], []
    for line in open(fname, 'r').readlines():
        x, y = line.strip().split('\t')
        xs.append(int(x))
        ys.append(float(y))
    return xs, ys


def perf():
    plt.figure(figsize=(12, 8))
    x, blas = extract(BLAS_FNAME)
    x, block = extract(BLOCK_FNAME)
    x, goto = extract(GOTO_FNAME)
    x, naive = extract(NAIVE_FNAME)
    x, naivet = extract(NAIVET_FNAME)

    sns.set_style('whitegrid')
    # plt.title('Performance')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('GFLOPS')
    ticks = [32, 64, 128, 256, 512, 1024]
    plt.xticks(ticks)
    sns.lineplot(x=x, y=blas, label='BLAS')
    sns.lineplot(x=x, y=block, label='Blocked')
    sns.lineplot(x=x, y=goto, label='GotoBLAS')
    sns.lineplot(x=x, y=naive, label='Naive')
    sns.lineplot(x=x, y=naivet, label='Naive Transposed')

    plt.savefig('perf.png')


def cache(folder, loc=None):
    plt.figure(figsize=(12, 8))

    sns.set_style('whitegrid')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('GFLOPS')
    ticks = [32, 64, 128, 256, 512, 1024]
    plt.xticks(ticks)

    lines = {}
    for fname in os.listdir(folder):
        params = tuple(map(int, re.findall('\d+', fname)))
        if loc:
            params = params[loc[0]] * params[loc[1]]
        x, lines[params] = extract(f'{folder}/{fname}')
    for size in sorted(lines.keys()):
        # print(len(x), len(lines[size]))
        sns.lineplot(x=x, y=lines[size], label=f'{int(size * 8 / 1024)} KB')

    plt.savefig(f'{os.path.basename(folder)}.png')


def avx(folder):
    plt.figure(figsize=(12, 8))

    sns.set_style('whitegrid')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('GFLOPS')
    ticks = [32, 64, 128, 256, 512, 1024]
    plt.xticks(ticks)

    lines = {}
    for fname in os.listdir(folder):
        label = fname.split('.')[0]
        x, lines[label] = extract(f'{folder}/{fname}')
    for label in lines.keys():
        # print(len(x), len(lines[size]))
        sns.lineplot(x=x, y=lines[label], label=f'{label}')

    plt.savefig(f'avx.png')


if __name__ == "__main__":
    avx(AVX)

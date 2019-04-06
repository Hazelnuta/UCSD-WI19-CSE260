import subprocess as sp
import sys
GCC_OPT = '-Ofast -ffast-math -funroll-loops -funroll-all-loops -mavx -mavx2 -mfma -ftree-vectorize -mfpmath=sse'


def goto():
    # goto (NC, KC, MC, MR, NR)
    testcases = [(512, 256, 128, 4, 8)]
    l3_tests = [(i, 256, 128, 4, 8) for i in range(128, 1025, 128)]
    l2_tests = [(512, 256, i, 4, 8) for i in range(32, 257, 32)]
    l1_tests = [(512, 256, 128, 4, i) for i in range(4, 17, 4)]
    for test in testcases:
        for avx in range(2):
            (NC, KC, MC, MR, NR) = test
            BLOCK_OPT = ' -D__GOTO__ -DNC=%d -DKC=%d -DMC=%d -DMR=%d -DNR=%d' % test
            if avx:
                BLOCK_OPT += ' -D__AVX256__'
            print('running', BLOCK_OPT)
            sp.call(['make', 'MY_OPT=' + GCC_OPT + BLOCK_OPT])
            filename = 'goto-' + \
                '-'.join(map(str, test)) + ('-avx' if avx else '') + '.result'
            sp.Popen(['./benchmark-blocked'],
                     stdout=open(filename, 'w')).communicate()


def level():
    # level (l2, l1, sm)
    testcases = [(256, 128, 32)]
    testcases = [(128, 32, i) for i in range(4, 33, 4)]
    for test in testcases:
        for avx in range(2):
            (l1, l2, sm) = test
            BLOCK_OPT = ' -D__LEVEL__ -Dblock_size_l2=%d -Dblock_size_l1=%d -Dblock_size_sm=%d' % test
            if avx:
                BLOCK_OPT += ' -D__AVX256__'
            print('running', BLOCK_OPT)
            sp.call(['make', 'MY_OPT=' + GCC_OPT + BLOCK_OPT])
            filename = 'level-' + \
                '-'.join(map(str, test)) + ('-avx' if avx else '') + '.result'
            sp.Popen(['./benchmark-blocked', '-c'],
                     stdout=open(filename, 'w')).communicate()


level()

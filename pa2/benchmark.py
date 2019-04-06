import subprocess as sp
tests = [j for i in range(32, 2048 + 1, 32) for j in range(i - 5, i + 6)]
# tests = tests[:tests.index(257)]
# tests = tests[tests.index(644) + 1:]
# print(tests)
for t in tests:
    sp.call(['srun', '--gres=gpu:1', '--partition=INTERACTIVE',
             './mmpy', '-R', '-r', '3', '-n', str(t)])

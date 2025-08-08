import os
import re

directory = './'

cmds = []
for filename in os.listdir(directory):
    pattern = re.compile(r'(st|svm)(_([a-z])+)+\.py')
    if not pattern.match(filename):
        continue
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        cmds.append('python ' + file_path)
cmd = ' && '.join(cmds)
os.system(cmd)
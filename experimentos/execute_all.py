import os
import re

directory = './'

for filename in os.listdir(directory):
    pattern = re.compile(r'(one)(_([a-z])+)+\.py')
    if not pattern.match(filename):
        continue
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        os.system('python ' + file_path)

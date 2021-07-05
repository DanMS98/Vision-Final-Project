import os
import subprocess

directory = os.fsencode(os.getcwd())
print(directory)
filenames = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".h264"):
        # print(filename)
        filename = 'file \'' + filename + '\''
        filenames.append(filename)

filenames.sort()
print(filenames)

with open('files.txt', 'w') as f:
    for name in filenames:
        f.write(name)
        f.write('\n')

subprocess.call(['ffmpeg', '-safe', '0', '-f', 'concat', '-i', 'files.txt', 'output2.mp4'])
# ffmpeg -safe 0 -f concat -i files.txt output.mp4

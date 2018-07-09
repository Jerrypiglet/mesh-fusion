import os
import sys
import math
import argparse
import numpy as np
from file_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert OFF to OBJ.')
    parser.add_argument('input', type=str, default='0_in', help='The input directory containing OFF files.')
    parser.add_argument('output', type=str, default='2_watertight', help='The output directory for OBJ files.')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('Input directory does not exist.')
        exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print('Created output directory.')
    else:
        print('Output directory exists; potentially overwriting contents.')


    for filename in os.listdir('original'):
        filepath = os.path.join('original', filename)
        if '.obj' in filepath and not(os.path.isfile(filepath.replace('.obj', '.off'))):
            print sys.path
            print(toGreen('%s is an .obj mesh. Converting to .off format...'%filepath))
            os.system('/Applications/meshlab.app/Contents/MacOS/meshlabserver -i %s -o %s'%(filepath, filepath.replace('.obj', '.off')))

    os.system('cp original/*.off 0_in/')

    print(toBlue('-- Rescaling..'))
    command_1_scale = 'python ../../1_scale.py --in_dir=0_in --out_dir=1_scaled'
    os.system(command_1_scale)
    print(toBlue('-- Rendering..'))
    command_2_render = 'python ../../2_fusion.py --mode=render --in_dir=1_scaled --depth_dir=2_depth --out_dir=2_watertight'
    os.system(command_2_render)
    print(toBlue('-- Fusing..'))
    command_2_fusion = 'python ../../2_fusion.py --mode=fuse --in_dir=1_scaled --depth_dir=2_depth --out_dir=2_watertight'
    os.system(command_2_fusion)
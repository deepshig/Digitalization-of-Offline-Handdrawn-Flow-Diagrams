# -*-coding:utf-8 -*-
'''
author:  hrwhipser
date   :  Jun 5, 2015
function : calculate feature and save in ./data/feature.txt ./data/tag.txt
'''
import os
import numpy as np
from handwriting_tools import read_image, to_the_same_size

sameSize = 16


def get_train_data(dataPath):
    train_input, desired_output = [], []
    i = 0
    for root, dirs, files in os.walk(dataPath):
        for _file in files:
            outCharacter = _file.split('_')[-2]
            suffix = _file.split('.')[-1]
            if outCharacter.isdigit() or suffix != 'bmp': continue
            test_img = to_the_same_size(read_image(dataPath + '\\' + _file))
            train_input.append(test_img.ravel())
            desired_output.append(outCharacter)
            print i, _file
            i += 1
            '''
            test = to_the_same_size(read_image(dataPath+'\\'+file))
            ax = plt.subplot(5,2,i+1)
            ax.set_axis_off()
            ax.imshow(test)
            train_input.append(test)
            '''
    return train_input, desired_output


if __name__ == '__main__':
    # dataPath = r'F:\handwritingData7'
    dataPath = r'E:\sd_nineteen\HSF_7'

    train_input, desired_output = get_train_data(dataPath)
    print len(train_input)
    with open('feature.txt', 'a+') as f:
        np.savetxt(f, train_input)
    with open('tag.txt', 'a+') as f:
        f.write('\n'.join(desired_output))
        f.write('\n')

    # dataPath = r'F:\handwritingData0'
    dataPath = r'E:\sd_nineteen\HSF_0'

    train_input, desired_output = get_train_data(dataPath)
    print len(train_input)
    with open('feature.txt', 'a+') as f:
        np.savetxt(f, train_input)
    with open('tag.txt', 'a+') as f:
        f.write('\n'.join(desired_output))
        f.write('\n')

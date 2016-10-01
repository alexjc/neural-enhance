#!/usr/bin/env python3
"""                          _              _                           
  _ __   ___ _   _ _ __ __ _| |   ___ _ __ | |__   __ _ _ __   ___ ___  
 | '_ \ / _ \ | | | '__/ _` | |  / _ \ '_ \| '_ \ / _` | '_ \ / __/ _ \ 
 | | | |  __/ |_| | | | (_| | | |  __/ | | | | | | (_| | | | | (_|  __/ 
 |_| |_|\___|\__,_|_|  \__,_|_|  \___|_| |_|_| |_|\__,_|_| |_|\___\___| 
                                                                        
"""
#
# Copyright (c) 2016, Alex J. Champandard.
#
# Neural Enhance is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General
# Public License version 3. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#

import os
import sys
import bz2
import glob
import math
import pickle
import random
import argparse
import itertools
import threading
import collections


# Configure all options first so we can later custom-load other libraries (Theano) based on device specified by user.
parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('--batch-size',         default=15, type=int)
add_arg('--batch-resolution',   default=128, type=int)
add_arg('--epoch-size',         default=72, type=int)
add_arg('--epochs',             default=100, type=int)
add_arg('--network-filters',    default=64, type=int)
add_arg('--perceptual-layer',   default='mse', type=str)
add_arg('--perceptual-weight',  default=1e0, type=float)
add_arg('--smoothness-weight',  default=0.0, type=float)
add_arg('--adversary-weight',   default=1e4, type=float)
add_arg('--scales',             default=1, type=int,            help='')
add_arg('--device',             default='gpu0', type=str,       help='Name of the CPU/GPU number to use, for Theano.')
args = parser.parse_args()


#----------------------------------------------------------------------------------------------------------------------

# Color coded output helps visualize the information a little better, plus it looks cool!
class ansi:
    BOLD = '\033[1;97m'
    WHITE = '\033[0;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[1;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'

def error(message, *lines):
    string = "\n{}ERROR: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(ansi.RED_B, ansi.RED, ansi.ENDC))
    sys.exit(-1)

def warn(message, *lines):
    string = "\n{}WARNING: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(ansi.YELLOW_B, ansi.YELLOW, ansi.ENDC))

print("""{}   {}Super Resolution for images and videos powered by Deep Learning!{}
  - Code licensed as AGPLv3, models under CC BY-NC-SA.{}""".format(ansi.CYAN_B, __doc__, ansi.CYAN, ansi.ENDC))

# Load the underlying deep learning libraries based on the device specified.  If you specify THEANO_FLAGS manually,
# the code assumes you know what you are doing and they are not overriden!
os.environ.setdefault('THEANO_FLAGS', 'floatX=float32,device={},force_device=True,allow_gc=True,'\
                                      'print_active_device=False'.format(args.device))

# Scientific & Imaging Libraries
import numpy as np
import scipy.optimize, scipy.ndimage, scipy.misc

# Numeric Computing (GPU)
import theano
import theano.tensor as T
import theano.tensor.nnet.neighbours

# Support ansi colors in Windows too.
if sys.platform == 'win32':
    import colorama

# Deep Learning Framework
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Deconv2DLayer as DeconvLayer, Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ConcatLayer, batch_norm

print('{}  - Using the device `{}` for neural computation.{}\n'.format(ansi.CYAN, theano.config.device, ansi.ENDC))


#----------------------------------------------------------------------------------------------------------------------
# Image Processing
#----------------------------------------------------------------------------------------------------------------------
class DataLoader(threading.Thread):

    def __init__(self):
        super(DataLoader, self).__init__(daemon=True)
        self.data_ready = threading.Event()
        self.data_copied = threading.Event()

        self.resolution = args.batch_resolution
        self.images = np.zeros((args.batch_size, 3, self.resolution, self.resolution), dtype=np.float32)

        self.cwd = os.getcwd()
        self.start()

    def run(self):
        files, cache = glob.glob('train/*.jpg'), {}

        while True:
            random.shuffle(files)

            for i, f in enumerate(files[:args.batch_size]):
                filename = os.path.join(self.cwd, f)
                try:
                    if f in cache:
                        img = cache[f]
                    else:
                        img = scipy.ndimage.imread(filename, mode='RGB')
                        cache[f] = img
                except Exception as e:
                    warn('Could not load `{}` as image.'.format(filename),
                         '  - Try fixing or removing the file before next run.')
                    files.remove(f)
                    continue

                if random.choice([True, False]):
                    img[:,:] = img[:,::-1]

                h = random.randint(0, img.shape[0] - self.resolution)
                w = random.randint(0, img.shape[1] - self.resolution)
                img = img[h:h+self.resolution, w:w+self.resolution]
                self.images[i] = np.transpose(img / 255.0 - 0.5, (2, 0, 1))

            self.data_ready.set()

            self.data_copied.wait()
            self.data_copied.clear()

    def copy(self, output):
        self.data_ready.wait()
        self.data_ready.clear()

        output[:] = self.images
        self.data_copied.set()


#----------------------------------------------------------------------------------------------------------------------
# Convolution Networks
#----------------------------------------------------------------------------------------------------------------------
class Model(object):

    def __init__(self):
        self.network = collections.OrderedDict()
        self.network['img'] = InputLayer((None, 3, None, None))
        self.network['img.scaled'] = PoolLayer(self.network['img'], pool_size=2**args.scales)

        self.setup_generator(self.network['img.scaled'])

        concatenated = lasagne.layers.ConcatLayer([self.network['img'], self.network['out']], axis=0)
        self.setup_perceptual(concatenated)
        self.load_perceptual()

        self.compile()

    def last_layer(self):
        return list(self.network.values())[-1]

    def setup_perceptual(self, input):
        """Use lasagne to create a network of convolution layers using pre-trained VGG19 weights.
        """

        offset = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1,3,1,1))
        self.network['percept'] = lasagne.layers.NonlinearityLayer(input, lambda x: ((x+0.5).clip(0.0, 1.0)*255.0) - offset)

        self.network['mse'] = self.network['percept']
        self.network['conv1_1'] = ConvLayer(self.network['percept'], 64, 3, pad=1)
        self.network['conv1_2'] = ConvLayer(self.network['conv1_1'], 64, 3, pad=1)
        self.network['pool1']   = PoolLayer(self.network['conv1_2'], 2, mode='max')
        self.network['conv2_1'] = ConvLayer(self.network['pool1'],   128, 3, pad=1)
        self.network['conv2_2'] = ConvLayer(self.network['conv2_1'], 128, 3, pad=1)
        self.network['pool2']   = PoolLayer(self.network['conv2_2'], 2, mode='max')
        self.network['conv3_1'] = ConvLayer(self.network['pool2'],   256, 3, pad=1)
        self.network['conv3_2'] = ConvLayer(self.network['conv3_1'], 256, 3, pad=1)
        self.network['conv3_3'] = ConvLayer(self.network['conv3_2'], 256, 3, pad=1)
        self.network['conv3_4'] = ConvLayer(self.network['conv3_3'], 256, 3, pad=1)
        self.network['pool3']   = PoolLayer(self.network['conv3_4'], 2, mode='max')
        self.network['conv4_1'] = ConvLayer(self.network['pool3'],   512, 3, pad=1)
        self.network['conv4_2'] = ConvLayer(self.network['conv4_1'], 512, 3, pad=1)
        self.network['conv4_3'] = ConvLayer(self.network['conv4_2'], 512, 3, pad=1)
        self.network['conv4_4'] = ConvLayer(self.network['conv4_3'], 512, 3, pad=1)
        self.network['pool4']   = PoolLayer(self.network['conv4_4'], 2, mode='max')
        self.network['conv5_1'] = ConvLayer(self.network['pool4'],   512, 3, pad=1)
        self.network['conv5_2'] = ConvLayer(self.network['conv5_1'], 512, 3, pad=1)
        self.network['conv5_3'] = ConvLayer(self.network['conv5_2'], 512, 3, pad=1)
        self.network['conv5_4'] = ConvLayer(self.network['conv5_3'], 512, 3, pad=1)

    def load_perceptual(self):
        """Open the serialized parameters from a pre-trained network, and load them into the model created.
        """
        vgg19_file = os.path.join(os.path.dirname(__file__), 'vgg19_conv.pkl.bz2')
        if not os.path.exists(vgg19_file):
            error("Model file with pre-trained convolution layers not found. Download here...",
                  "https://github.com/alexjc/neural-doodle/releases/download/v0.0/vgg19_conv.pkl.bz2")

        data = pickle.load(bz2.open(vgg19_file, 'rb'))
        layers = lasagne.layers.get_all_layers(self.last_layer(), treat_as_input=[self.network['percept']])
        for p, d in zip(itertools.chain(*[l.get_params() for l in layers]), data): p.set_value(d)

    def setup_generator(self, input):
        f = args.network_filters
        self.network['iter.0'] = ConvLayer(input, f, filter_size=(1,1), stride=(1,1), pad=0,)

        for i in range(args.scales, 0, -1):
            self.network['scale%i.2'%i] = DeconvLayer(self.last_layer(), f, filter_size=(4,4), stride=(2,2), crop=1)
            self.network['scale%i.1'%i] = ConvLayer(self.network['scale%i.2'%i], f, filter_size=(3,3), pad=1)

        self.network['out'] = ConvLayer(self.last_layer(), 3, filter_size=(1,1), stride=(1,1), pad=0, b=None,
                                                              nonlinearity=lasagne.nonlinearities.tanh)

    def compile(self):
        self.learning_rate = theano.shared(np.array(1e-4, dtype=theano.config.floatX))

        input_tensor = T.tensor4()
        output_layers = [self.network['out'], self.network[args.perceptual_layer]]
        input_layers = {self.network['img']: input_tensor}

        gen_out, percept_out = lasagne.layers.get_output(output_layers, input_layers, deterministic=False)
        losses = [self.loss_perceptual(percept_out) * args.perceptual_weight,
                  self.loss_total_variation(gen_out) * args.smoothness_weight]

        params = lasagne.layers.get_all_params(self.network['out'], trainable=True)
        updates = lasagne.updates.adam(sum(losses, 0.0), params, learning_rate=self.learning_rate)
        self.fit = theano.function([input_tensor], losses, updates=updates)

        gen_out, gen_inp = lasagne.layers.get_output([self.network['out'], self.network['img']],
                                                     input_layers, deterministic=True)
        self.predict = theano.function([input_tensor], [gen_out, gen_inp])

    def loss_perceptual(self, p):
        return lasagne.objectives.squared_error(p[:args.batch_size], p[args.batch_size:]).mean()
    
    def loss_total_variation(self, x):
        return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).mean()
 

class NeuralEnhancer(object):

    def __init__(self):
        self.thread = DataLoader()
        self.model = Model()

    def imsave(self, fn, img):
        img = np.transpose(img + 0.5, (1, 2, 0)).clip(0.0, 1.0)
        image = scipy.misc.toimage(img * 255.0, cmin=0, cmax=255)
        image.save(fn)

    def show_progress(self, repro, orign):
        for i in range(args.batch_size):
            self.imsave('test/%03i_orign.png' % i, orign[i])
            self.imsave('test/%03i_repro.png' % i, repro[i])

    def train(self):
        print('\n{}Training {} epochs with batch size {}.{}'\
                .format(ansi.BLUE_B, args.epochs, args.batch_size, ansi.ENDC))

        images = np.zeros((args.batch_size, 3, args.batch_resolution, args.batch_resolution), dtype=np.float32)

        l_min, l_max, l_mult = 1E-7, 1E-3, 0.2
        t_cur, t_i, t_mult = 0, 150, 1

        i, last, running = 0, float('inf'), None
        for _ in range(args.epochs):
            total = 0.0
            for _ in range(args.epoch_size):
                i += 1
                l_r = l_min + 0.5 * (l_max - l_min) * (1.0 + math.cos(t_cur / t_i * math.pi))
                t_cur += 1
                self.model.learning_rate.set_value(l_r)

                if t_cur >= t_i:
                    t_cur = 0
                    t_i = int(t_i * t_mult)
                    l_max = max(l_max * l_mult, 1e-10)
                    l_min = max(l_min * l_mult, 1e-6)
                
                self.thread.copy(images)
                losses = self.model.fit(images)
                l = sum(losses)
                total += l
                running = l if running is None else running * 0.9 + 0.1 * l

                print('↑' if l >= running else '↓', end='', flush=True)

            self.show_progress(*self.model.predict(images))
            last = total / args.epoch_size
            print('\nLosses total:', last)


if __name__ == "__main__":
    enhancer = NeuralEnhancer()
    try:
        enhancer.train()
    except KeyboardInterrupt:
        pass

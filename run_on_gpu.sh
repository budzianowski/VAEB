#!/bin/bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python VAEB.py --continuous 1

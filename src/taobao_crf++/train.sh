#!/bin/bash
crf_learn -f 4 -p 4 -t -c 3 template crf_train.data models > crf_train.rst
#crf_test -m models crf_test.data > crf_test.rst

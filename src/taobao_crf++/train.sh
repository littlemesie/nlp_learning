#!/bin/bash
crf_learn -f 4 -p 4 -t -c 3 template crf_train.data model > crf_train.rst
#crf_test -m model crf_test.data > crf_test.rst

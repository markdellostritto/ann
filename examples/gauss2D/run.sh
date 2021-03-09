#!/bin/bash
py=python
nn_fit=../../bin/nn_fit.exe
$py write_gauss.py
$nn_fit nn_fit_2D.txt > nn_fit_2D.out


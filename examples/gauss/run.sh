#!/bin/bash
py=python
nn_fit=../../bin/nn_fit_1D.exe
$py write_gauss.py
$nn_fit -in nn_fit_1D.txt -out nn_fit_1D.out


#!/bin/bash

set -exu

pushd src/evaluation/
rm special_partition.c*
cythonize -a -i special_partition.pyx
popd

#!/bin/bash

set -exu

pushd src/evaluation/
rm special_partition.c*
cythonize -a -i special_partition.pyx
rm special_hac.c*
cythonize -a -i special_hac.pyx
popd

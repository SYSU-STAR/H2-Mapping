#!/bin/bash

cd third_party/sparse_octree
python setup.py install

cd ../sparse_voxels
python setup.py install

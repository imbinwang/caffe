#!/usr/bin/env sh

TOOLS=./build/tools/

$TOOLS/caffe train --solver=examples/triplet_LINEMOD/3d_triplet_solver_1.prototxt --gpu=8

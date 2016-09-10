#!/usr/bin/env sh

TOOLS=./build/tools/

cp -r ./examples/triplet_LINEMOD/linemod_triplet_train_leveldb ./examples/triplet_LINEMOD/linemod_triplet_train_leveldb_copy
cp -r ./examples/triplet_LINEMOD/linemod_triplet_test_leveldb ./examples/triplet_LINEMOD/linemod_triplet_test_leveldb_copy

$TOOLS/caffe train --solver=examples/triplet_LINEMOD/3d_triplet_solver_2.prototxt --gpu=9

#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=./build/examples/triplet_LINEMOD
DATA=~/wb/LINEMOD

echo "Creating leveldb for LINEMOD triplet training and test..."

rm -rf ./examples/triplet_LINEMOD/linemod_triplet_train_leveldb
rm -rf ./examples/triplet_LINEMOD/linemod_triplet_test_leveldb

$EXAMPLES/convert_3d_triplet_data.bin \
    $DATA/_train/binary_image \
    $DATA/_train/binary_label \
    ./examples/triplet_LINEMOD/linemod_triplet_train_leveldb \
    1
$EXAMPLES/convert_3d_triplet_data.bin \
    $DATA/_test/binary_image \
    $DATA/_test/binary_label \
    ./examples/triplet_LINEMOD/linemod_triplet_test_leveldb \
    1
echo "Done."

// Usage:
// convert_3d_data input_image_file input_label_file output_db_file
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "math.h"
#include "stdint.h"

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void read_image(std::ifstream* image_file, std::ifstream* label_file,
        uint32_t index, uint32_t rows, uint32_t cols,
        char* pixels, int* label_id_temp, float* label_pos_temp, int* label_id, float* label_pos, int rgb_use) {
  if (rgb_use == 0) {
    image_file->seekg(index * rows * cols + 16);
    image_file->read(pixels, rows * cols);
    label_file->seekg(index * 16 + 8);
    label_file->read(reinterpret_cast<char*>(label_id_temp), 4);
    label_file->read(reinterpret_cast<char*>(label_pos_temp), 12);
    for (int i=0; i<4; ++i) {
      if (i==0){
        int *tmp = reinterpret_cast<int*>(label_id_temp);
        *label_id = *tmp;
      }
      else{
        float *tmp = reinterpret_cast<float*>(label_pos_temp);
        label_pos[i-1] = tmp[i-1];
      }
    }
  } else {
    image_file->seekg(3 * index * rows * cols + 16);
    image_file->read(pixels, 3 * rows * cols);
    label_file->seekg(index * 16 + 8);
    label_file->read(reinterpret_cast<char*>(label_id_temp), 4);
    label_file->read(reinterpret_cast<char*>(label_pos_temp), 12);
    for (int i=0; i<4; ++i) {
      if (i==0){
        int *tmp = reinterpret_cast<int*>(label_id_temp);
        *label_id = *tmp;
      }
      else{
        float *tmp = reinterpret_cast<float*>(label_pos_temp);
        label_pos[i-1] = tmp[i-1];
      }
    }
  }
}

void convert_dataset(const char* image_filename, const char* label_filename,
        const char* db_filename, const char* rgb_use) {
  int rgb_use1 = atoi(rgb_use);
  // Open files
  std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
  std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
  CHECK(image_file) << "Unable to open file " << image_filename;
  CHECK(label_file) << "Unable to open file " << label_filename;
  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  image_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
  label_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2050) << "Incorrect label file magic.";
  image_file.read(reinterpret_cast<char*>(&num_items), 4);
  num_items = swap_endian(num_items);
  label_file.read(reinterpret_cast<char*>(&num_labels), 4);
  num_labels = swap_endian(num_labels);
  CHECK_EQ(num_items, num_labels);
  image_file.read(reinterpret_cast<char*>(&rows), 4);
  rows = swap_endian(rows);
  image_file.read(reinterpret_cast<char*>(&cols), 4);
  cols = swap_endian(cols);

  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";

  int* label_id_temp = new int;
  float* label_pos_temp = new float[3];
  int *label_i = new int;
  float *label_i_pos = new float[3];
  int *label_j = new int;
  float *label_j_pos = new float[3];
  int *label_k = new int;
  float *label_k_pos = new float[3];
  int *label_l = new int;
  float *label_l_pos = new float[3];
  int *label_m = new int;
  float *label_m_pos = new float[3];

  int db_size;
  if (rgb_use1 == 0)
    db_size = rows * cols;
  else
    db_size = 3 * rows * cols;
  char* pixels1 = new char[db_size];
  char* pixels2 = new char[db_size];
  char* pixels3 = new char[db_size];
  char* pixels4 = new char[db_size];
  char* pixels5 = new char[db_size];
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  std::string value;
  caffe::Datum datum;
  if (rgb_use1 == 0)
    datum.set_channels(1);
  else
    datum.set_channels(3);
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;

  std::vector<int> idx_vec;
  for (int i=0; i<num_items; ++i){
    idx_vec.push_back(i);
  }
  // iteration in the samples of all class
  int counter = 0;
  for (int times = 0; times < 10; ++times) {
    caffe::shuffle(idx_vec.begin(), idx_vec.end());
    for (int ind = 0; ind < int(num_items); ++ind) {
      int i = idx_vec[ind];
      int j = caffe::caffe_rng_rand() % num_items;  // pick triplet groups
      int k = caffe::caffe_rng_rand() % num_items;
      int l = caffe::caffe_rng_rand() % num_items;  // pick pair wise groups
      int m = caffe::caffe_rng_rand() % num_items;
      read_image(&image_file, &label_file, i, rows, cols,  // read triplet
        pixels1, label_id_temp,label_pos_temp, label_i, label_i_pos, rgb_use1);
      read_image(&image_file, &label_file, j, rows, cols,
        pixels2, label_id_temp,label_pos_temp, label_j, label_j_pos, rgb_use1);
      read_image(&image_file, &label_file, k, rows, cols,
        pixels3, label_id_temp,label_pos_temp, label_k, label_k_pos, rgb_use1);
      read_image(&image_file, &label_file, l, rows, cols,  // read pair wise
        pixels4, label_id_temp,label_pos_temp, label_l, label_l_pos, rgb_use1);
      read_image(&image_file, &label_file, m, rows, cols,
        pixels5, label_id_temp,label_pos_temp, label_m, label_m_pos, rgb_use1);

      bool pair_pass = false;
      bool triplet1_pass = false;
      bool triplet2_pass = false;
      bool triplet3_class_same = false;
      bool triplet3_pass = false;

      float ij_diff_x =label_i_pos[0]-label_j_pos[0];
      float ij_diff_y =label_i_pos[1]-label_j_pos[1];
      float ij_diff_z = label_i_pos[2]-label_j_pos[2];
      float im_diff_x = label_i_pos[0]-label_m_pos[0];
      float im_diff_y = label_i_pos[1]-label_m_pos[1];
      float im_diff_z = label_i_pos[2]-label_m_pos[2];

      float ij_x = ij_diff_x*ij_diff_x;
      float ij_y = ij_diff_y*ij_diff_y;
      float ij_z = ij_diff_z*ij_diff_z;
      float im_x = im_diff_x*im_diff_x;
      float im_y = im_diff_y*im_diff_y;
      float im_z = im_diff_z*im_diff_z;

      float dist_ij = std::sqrt(ij_x + ij_y + ij_z);
      float dist_im = std::sqrt(im_x + im_y + im_z);
      
      if (*label_i == *label_j && dist_ij < 100/3 && dist_ij != 0)
        pair_pass = true;
      if (pair_pass && (*label_i  != *label_k))
        triplet1_pass = true;
      if (pair_pass && (*label_i  != *label_l))
        triplet2_pass = true;
      if (pair_pass && (*label_i  == *label_m))
        triplet3_class_same = true;
      if (triplet3_class_same && dist_im > 100/3)
        triplet3_pass = true;
      if (pair_pass && triplet1_pass && triplet2_pass && triplet3_pass) {
        datum.set_data(pixels1, db_size);  // set data
        datum.set_label(*label_i);
        datum.SerializeToString(&value);
        snprintf(key, kMaxKeyLength, "%08d", counter);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
        counter++;
        datum.set_data(pixels2, db_size);  // set data
        datum.set_label(*label_j);
        datum.SerializeToString(&value);
        snprintf(key, kMaxKeyLength, "%08d", counter);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
        counter++;
        datum.set_data(pixels3, db_size);  // set data
        datum.set_label(*label_k);
        datum.SerializeToString(&value);
        snprintf(key, kMaxKeyLength, "%08d", counter);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
        counter++;
        datum.set_data(pixels4, db_size);  // set data
        datum.set_label(*label_l);
        datum.SerializeToString(&value);
        snprintf(key, kMaxKeyLength, "%08d", counter);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
        counter++;
        datum.set_data(pixels5, db_size);  // set data
        datum.set_label(*label_m);
        datum.SerializeToString(&value);
        snprintf(key, kMaxKeyLength, "%08d", counter);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
        counter++;
      } else {
        --ind;
      }
    }  // iteration in the samples of all class
  }  // iteration in times
  delete db;
  delete pixels1;
  delete pixels2;
  delete pixels3;
  delete pixels4;
  delete pixels5;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    printf("This script converts the dataset to the leveldb format used\n"
           "by caffe to train a triplet network.\n"
           "Usage:\n"
           "    convert_3d_data input_image_file input_label_file "
           "output_db_file rgb_use \n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(argv[1], argv[2], argv[3], argv[4]);
  }
  return 0;
}

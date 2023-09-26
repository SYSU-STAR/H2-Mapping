#include <memory>
#include <torch/script.h>
#include <torch/custom_class.h>

enum OcType
        {
    NONLEAF = -1,
    SURFACE = 0,
    FEATURE = 1
        };

class Octant : public torch::CustomClassHolder
        {
        public:
            inline Octant()
            {

                code_ = 0;
                side_ = 0;
                index_ = next_index_++;
                depth_ = -1;
                is_leaf_ = false;
                children_mask_ = 0;
                type_ = NONLEAF;
                for (unsigned int i = 0; i < 8; i++)
                {
                    child_ptr_[i] = nullptr;
                    // feature_index_[i] = -1;
                }
            }
            ~Octant() {}

            // std::shared_ptr<Octant> &child(const int x, const int y, const int z)
            // {
            //     return child_ptr_[x + y * 2 + z * 4];
            // };

            // std::shared_ptr<Octant> &child(const int offset)
            // {
            //     return child_ptr_[offset];
            // }
            Octant *&child(const int x, const int y, const int z)
            {
                return child_ptr_[x + y * 2 + z * 4];
            };

            Octant *&child(const int offset)
            {
                return child_ptr_[offset];
            }

            void set_leaf_index_(int leaf_type){
                if (leaf_type != NONLEAF ){
                    leaf_index_ = leaf_next_index_++;
                }
            }

            uint64_t code_;
            bool is_leaf_;
            unsigned int side_;
            unsigned char children_mask_;
            // std::shared_ptr<Octant> child_ptr_[8];
            // int feature_index_[8];
            int index_;
            int depth_;
            int type_;
            // int feat_index_;
            Octant *child_ptr_[8];
            static int next_index_;
            int leaf_index_;
            static int leaf_next_index_;
        };

template <typename T>
class MyMat {
public:
    MyMat() = default;

    void resize(int rows, int cols) {
        rows_ = rows;
        cols_ = cols;
        ptr_ = new T[rows_ * cols_];
    }

    ~MyMat() {
        delete ptr_;
    }

    T at(int row, int col) {
        return ((T*)ptr_)[row * cols_ + col];
    }

    void set(int row, int col, T val) {
        ((T*)ptr_)[row * cols_ + col] = val;
    }

    T* ptr() {
        return (T*)ptr_;
    }

private:
    int rows_;
    int cols_;
    void *ptr_;
};



class Octree : public torch::CustomClassHolder
        {
        public:
            Octree();
            ~Octree();
            void init(int64_t grid_dim, int64_t grid_num, double voxel_size);

            // allocate voxels
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> insert(torch::Tensor vox);

            // find a particular octant
            Octant *find_octant(std::vector<float> coord);

            // test intersections
            bool has_voxel(torch::Tensor pose);

            // query features
            torch::Tensor get_features(torch::Tensor pts);

            // get all voxels
            torch::Tensor get_voxels();
            std::vector<float> get_voxel_recursive(Octant *n);

            // get leaf voxels
            torch::Tensor get_leaf_voxels();
            std::vector<float> get_leaf_voxel_recursive(Octant *n);

            // count nodes
            int64_t count_nodes();
            int64_t count_recursive(Octant *n);

            // count leaf nodes
            int64_t count_leaf_nodes();
            // int64_t leaves_count_recursive(std::shared_ptr<Octant> n);
            int64_t leaves_count_recursive(Octant *n);

        public:
            int size_;
            int grid_num_;
            int max_level_;

//            float children_[1000000][8];
//            float voxel_[1000000][4];
//            int features_[1000000][8];
//            int vox_has_value[1000000][1];

            MyMat<float> children_;
            MyMat<float> voxel_;
            MyMat<int> features_;
            MyMat<int> vox_has_value;

//            float **children_;
//            float **voxel_;
//            int **features_;
//            int **vox_has_value;


            // temporal solution
            double voxel_size_;
            std::vector<torch::Tensor> all_pts;

        private:
            std::set<uint64_t> all_keys;
            std::map<uint64_t, int> surface_keys;

            // std::shared_ptr<Octant> root_;
            Octant *root_;
            // static int feature_index;

            // internal count function
            std::pair<int64_t, int64_t> count_nodes_internal();
            std::pair<int64_t, int64_t> count_recursive_internal(Octant *n);


        };
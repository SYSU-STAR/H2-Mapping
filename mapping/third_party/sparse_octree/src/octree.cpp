#include "octree.h"
#include "utils.h"
#include <queue>
#include <iostream>

// #define MAX_HIT_VOXELS 10
// #define MAX_NUM_VOXELS 10000

int Octant::next_index_ = 0;
int Octant::leaf_next_index_ = 0;
// int Octree::feature_index = 0;

int incr_x[8] = {0, 0, 0, 0, 1, 1, 1, 1};
int incr_y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
int incr_z[8] = {0, 1, 0, 1, 0, 1, 0, 1};

Octree::Octree()
{
}

Octree::~Octree()
{
}

void Octree::init(int64_t grid_dim, int64_t grid_num, double voxel_size)
{
    size_ = grid_dim;
    voxel_size_ = voxel_size;
    max_level_ = log2(size_);
    // root_ = std::make_shared<Octant>();
    root_ = new Octant();
    root_->side_ = size_;
    // root_->depth_ = 0;
    root_->is_leaf_ = false;

    grid_num_ = grid_num;

    children_.resize(grid_num, 8);
    features_.resize(grid_num, 8);
    voxel_.resize(grid_num, 4);
    vox_has_value.resize(grid_num, 1);

    for (int i = 0; i <grid_num_; i ++ ){
        for (int j = 0; j < 8; j ++){
            if (j < 4)
                voxel_.set(i, j, 0.0);
            if (j < 1)
                vox_has_value.set(i, j, 0);
            children_.set(i, j, -1.0);
            features_.set(i, j, -1);
        }
    }
    vox_has_value.set(root_->index_, 0, 1);
    auto xyz = decode(root_->code_);
    voxel_.set(root_->index_, 0, float(xyz[0]));
    voxel_.set(root_->index_, 1, float(xyz[1]));
    voxel_.set(root_->index_, 2, float(xyz[2]));
    voxel_.set(root_->index_, 3, float(root_->side_));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor ,torch::Tensor ,torch::Tensor>  Octree::insert(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 2>();
    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        //        return;
    }

    int frame_voxel_idx[points.size(0)][1];
    for (int i = 0; i < points.size(0); ++i)
    {
        std::vector<int> features;
        std::vector<int> features_id;
        bool is_surface = false;
        for (int j = 0; j < 8; ++j)
            {
            int x = points[i][0] + incr_x[j];
            int y = points[i][1] + incr_y[j];
            int z = points[i][2] + incr_z[j];
            uint64_t key = encode(x, y, z);

            int surface_id;

//            all_keys.insert(key);
            auto l_it = surface_keys.find(key);
            if (l_it != surface_keys.end() && j==0)
            {
                frame_voxel_idx[i][0] = l_it->second;
                continue;
            }
            const unsigned int shift = MAX_BITS - max_level_ - 1;

            auto n = root_;
            unsigned edge = size_ / 2;

            for (int d = 1; d <= max_level_; edge /= 2, ++d)
            {
                const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0); // current point belongs to which child 0-8
                const int parentid = n->index_;
                auto tmp = n->child(childid);
                if (!tmp)
                {
                    const uint64_t code = key & MASK[d + shift];
                    const bool is_leaf = (d == max_level_); // the leaf node only exist in the deepest level
                    int tmp_type;
                    tmp_type = is_leaf ? (j == 0 ? SURFACE : FEATURE) : NONLEAF;
                    tmp = new Octant();
                    tmp->set_leaf_index_(tmp_type);
                    tmp->code_ = code;
                    tmp->side_ = edge;
                    tmp->is_leaf_ = is_leaf;
                    tmp->type_ = tmp_type;

                    n->children_mask_ = n->children_mask_ | (1 << childid);
                    n->child(childid) = tmp;
                }
                else
                {
                    if (tmp->type_ == FEATURE && j == 0)
                        tmp->type_ = SURFACE;
                }
                vox_has_value.set(tmp->index_, 0, 1);
                if (tmp->type_ != FEATURE){
                    children_.set(parentid, childid, tmp->index_);
                }
                auto xyz = decode(tmp->code_);

                if (tmp->type_ != FEATURE){
                    voxel_.set(tmp->index_, 0, xyz[0]);
                    voxel_.set(tmp->index_, 1, xyz[1]);
                    voxel_.set(tmp->index_, 2, xyz[2]);
                    voxel_.set(tmp->index_, 3, float(tmp->side_));
                }
                if (is_surface && tmp->is_leaf_){
                    features_.set(surface_id, j, tmp->leaf_index_);
                }
                if (tmp->type_ == SURFACE && j == 0){
                    is_surface = true;
                    surface_id = tmp->index_;
                    features_.set(surface_id, j, tmp->leaf_index_);
                    frame_voxel_idx[i][0] = tmp->leaf_index_;
                    surface_keys.insert(std::pair < uint64_t, int > (key,tmp->leaf_index_));
                }
                n = tmp;
            }
            }
    }
    return std::make_tuple(torch::from_blob(voxel_.ptr(), {grid_num_,4}, dtype(torch::kFloat32)).clone(),
                           torch::from_blob(children_.ptr(), {grid_num_,8}, dtype(torch::kFloat32)).clone(),
                           torch::from_blob(features_.ptr(), {grid_num_,8}, dtype(torch::kInt32)).clone(),
                           torch::from_blob(vox_has_value.ptr(), {grid_num_,1}, dtype(torch::kInt32)).clone(),
                           torch::from_blob(frame_voxel_idx, {points.size(0),1}, dtype(torch::kInt32)).clone());
}

Octant *Octree::find_octant(std::vector<float> coord)
{
    int x = int(coord[0]);
    int y = int(coord[1]);
    int z = int(coord[2]);
    // uint64_t key = encode(x, y, z);
    // const unsigned int shift = MAX_BITS - max_level_ - 1;

    auto n = root_;
    unsigned edge = size_ / 2;
    for (int d = 1; d <= max_level_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n->child(childid);
        if (!tmp)
            return nullptr;

        n = tmp;
    }
    return n;
}

bool Octree::has_voxel(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 1>();
    if (points.size(0) != 3)
    {
        return false;
    }

    int x = int(points[0]);
    int y = int(points[1]);
    int z = int(points[2]);

    auto n = root_;
    unsigned edge = size_ / 2;
    for (int d = 1; d <= max_level_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n->child(childid);
        if (!tmp)
            return false;

        n = tmp;
    }

    if (!n)
        return false;
    else
        return true;
}

torch::Tensor Octree::get_features(torch::Tensor pts)
{
}

torch::Tensor Octree::get_leaf_voxels()
{
    std::vector<float> voxel_coords = get_leaf_voxel_recursive(root_);

    int N = voxel_coords.size() / 3;
    torch::Tensor voxels = torch::from_blob(voxel_coords.data(), {N, 3});
    return voxels.clone();
}

std::vector<float> Octree::get_leaf_voxel_recursive(Octant *n)
{
    if (!n)
        return std::vector<float>();

    if (n->is_leaf_ && n->type_ == SURFACE)
    {
        auto xyz = decode(n->code_);
        return {xyz[0], xyz[1], xyz[2]};
    }

    std::vector<float> coords;
    for (int i = 0; i < 8; i++)
    {
        auto temp = get_leaf_voxel_recursive(n->child(i));
        coords.insert(coords.end(), temp.begin(), temp.end());
    }

    return coords;
}

torch::Tensor Octree::get_voxels()
{
    std::vector<float> voxel_coords = get_voxel_recursive(root_);
    int N = voxel_coords.size() / 4;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor voxels = torch::from_blob(voxel_coords.data(), {N, 4}, options);
    return voxels.clone();
}

std::vector<float> Octree::get_voxel_recursive(Octant *n)
{
    if (!n)
        return std::vector<float>();

    auto xyz = decode(n->code_);
    std::vector<float> coords = {xyz[0], xyz[1], xyz[2], float(n->side_)};
    for (int i = 0; i < 8; i++)
    {
        auto temp = get_voxel_recursive(n->child(i));
        coords.insert(coords.end(), temp.begin(), temp.end());
    }

    return coords;
}

std::pair<int64_t, int64_t> Octree::count_nodes_internal()
{
    return count_recursive_internal(root_);
}

std::pair<int64_t, int64_t> Octree::count_recursive_internal(Octant *n)
{
    if (!n)
        return std::make_pair<int64_t, int64_t>(0, 0);

    if (n->is_leaf_)
        return std::make_pair<int64_t, int64_t>(1, 1);

    auto sum = std::make_pair<int64_t, int64_t>(1, 0);

    for (int i = 0; i < 8; i++)
    {
        auto temp = count_recursive_internal(n->child(i));
        sum.first += temp.first;
        sum.second += temp.second;
    }

    return sum;
}

int64_t Octree::count_nodes()
{
    return count_recursive(root_);
}

int64_t Octree::count_recursive(Octant *n)
{
    if (!n)
        return 0;

    int64_t sum = 1;

    for (int i = 0; i < 8; i++)
    {
        sum += count_recursive(n->child(i));
    }

    return sum;
}

int64_t Octree::count_leaf_nodes()
{
    return leaves_count_recursive(root_);
}

// int64_t Octree::leaves_count_recursive(std::shared_ptr<Octant> n)
int64_t Octree::leaves_count_recursive(Octant *n)
{
    if (!n)
        return 0;

    if (n->type_ == SURFACE)
    {
        return 1;
    }

    int64_t sum = 0;

    for (int i = 0; i < 8; i++)
    {
        sum += leaves_count_recursive(n->child(i));
    }

    return sum;
}

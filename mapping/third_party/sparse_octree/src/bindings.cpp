#include "octree.h"
#include "test.h"

TORCH_LIBRARY(svo, m)
{
    m.def("encode", &encode_torch);

    m.class_<Octant>("Octant")
        .def(torch::init<>());

    m.class_<Octree>("Octree")
        .def(torch::init<>())
        .def("init", &Octree::init)
        .def("insert", &Octree::insert)
        .def("get_voxels", &Octree::get_voxels)
        .def("get_leaf_voxels", &Octree::get_leaf_voxels)
        .def("get_features", &Octree::get_features)
        .def("count_nodes", &Octree::count_nodes)
        .def("count_leaf_nodes", &Octree::count_leaf_nodes)
        .def("has_voxel", &Octree::has_voxel)
        ;
}
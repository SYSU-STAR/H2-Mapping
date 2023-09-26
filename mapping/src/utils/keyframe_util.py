import torch

from torch.nn.utils.rnn import pad_sequence

"""
    Overwrite all voxels contained in the keyframe multiple times
    
    Note that there is a related implementation outside of this function. 
    In the insert_keyframe class method in mapping, The function is to add the new voxels contained 
    in each newly added key frame to the set of unoptimized voxels. 
    Cover all the voxels contained in the key frame multiple times
    Args:
        kf_graph (list, l): Save information about keyframes into a list.
        kf_seen_voxel_num (list, l): This is a list, each element is the number of the corresponding 
                                    voxels contained in the keyframe in kf_graph
        unoptimized_voxels (tensor, N*3):  Coordinates of all unoptimized voxels  
        optimized_voxels (tensor, M*3): Coordinates of all optimized voxels
        windows_size (int, w): Number of keyframe pools
    Returns:
        target_graph (list, w): The keyframe of the final selected w
        unoptimized_voxels (tensor, N'*3): The voxel seen through the selected keyframe, 
                                        the updated unoptimized voxel coordinates
        optimized_voxels (tensor, M'*3): Voxels seen through selected keyframes, 
                                    updated optimized voxel coordinates
                
"""


def multiple_max_set_coverage(kf_graph, kf_seen_voxel_num, kf_unoptimized_voxels, kf_optimized_voxels,
                              windows_size, kf_svo_idx, kf_all_voxels, num_vertexes):
    cnt = 0
    target_graph = []
    padded_tensor = pad_sequence(kf_svo_idx, batch_first=True, padding_value=-1)[:, :, 0]
    if kf_unoptimized_voxels is None:
        kf_unoptimized_voxels = torch.zeros(num_vertexes).cuda().bool()  # unoptimized voxels
        kf_all_voxels = torch.zeros(num_vertexes).cuda().bool()  # All voxels to be optimized
        kf_optimized_voxels = torch.zeros(
            num_vertexes).cuda().bool()  # The voxels seen by the currently selected x keyframes

        kf_seen_voxel_num = torch.tensor(kf_seen_voxel_num)  # (N)
        kf_unoptimized_voxels[padded_tensor.long() + 1] += True  # Empty number 0, because the back pad is filled with 0
        kf_unoptimized_voxels[0] = False
        value, index = torch.max(kf_seen_voxel_num, dim=0)
        target_graph += [kf_graph[index]]
        kf_unoptimized_voxels[kf_svo_idx[index].long() + 1] *= False  # Empty number 0
        kf_optimized_voxels[kf_svo_idx[index].long() + 1] += True
        cnt += 1
    while cnt != int(windows_size):
        result_num = kf_unoptimized_voxels[padded_tensor.long() + 1].sum(-1)  # Empty number 0
        value, index = torch.max(result_num, dim=0)
        target_graph += [kf_graph[index]]
        kf_unoptimized_voxels[kf_svo_idx[index].long() + 1] *= False
        kf_optimized_voxels[kf_svo_idx[index].long() + 1] += True
        cnt += 1
        if kf_unoptimized_voxels.any() == False:  # If all are optimized
            kf_all_voxels[padded_tensor.long() + 1] += True
            kf_all_voxels[0] = False  # Empty number 0
            # Unoptimized voxels = all voxels that need to be optimized - \
            # voxels seen by the currently selected x keyframes
            kf_unoptimized_voxels = kf_all_voxels * ~kf_optimized_voxels
        kf_optimized_voxels *= False
    return target_graph, kf_unoptimized_voxels, kf_optimized_voxels, kf_all_voxels

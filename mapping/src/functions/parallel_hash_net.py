import tinycudann as tcnn
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
                 bound=None,
                 voxel_size=None,
                 L=None,
                 F_entry=None,
                 log2_T=None,
                 b=None,
                 **kwargs):
        super().__init__()

        self.bound = torch.FloatTensor(bound)
        self.bound_dis = self.bound[:, 1] - self.bound[:, 0]
        self.max_dis = torch.ceil(torch.max(self.bound_dis))
        N_min = int(self.max_dis / voxel_size)
        self.hash_sdf_out = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=1,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F_entry,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,  # 1/base_resolution is the grid_size
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_la2yers": 1,
                }
            )

        self.hash_color_out = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=3,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F_entry,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,  # 1/base_resolution is the grid_size
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )

    def get_color(self, xyz):
        # xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.max_dis.to(xyz.device)
        xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.bound_dis.to(xyz.device)
        rgb = self.hash_color_out(xyz)
        return rgb

    def get_sdf(self, xyz):
        # xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.max_dis.to(xyz.device)
        xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.bound_dis.to(xyz.device)
        sdf = self.hash_sdf_out(xyz)
        return sdf

    def get_values(self, xyz):
        # xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.max_dis.to(xyz.device)
        xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.bound_dis.to(xyz.device)
        sdf = self.hash_sdf_out(xyz)
        rgb = self.hash_color_out(xyz)
        # outputs = torch.cat([rgb, sdf], dim=-1)

        return sdf, rgb

    def forward(self, xyz):
        sdf, rgb = self.get_values(xyz)
        return {
            'color': rgb[:, :3],
            'sdf': sdf[:, 0]
        }


if __name__ == "__main__":
    network = Decoder(1, 128, 16, skips=[], embedder='none', multires=0)
    print(network)

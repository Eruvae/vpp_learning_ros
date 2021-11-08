try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
import torch.utils.data
import MinkowskiEngine as ME


class VAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, sinput, gt_target):
        means, log_vars = self.encoder(sinput)
        zs = means
        if self.training:
            zs = zs + torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F)
        out_cls, targets, sout = self.decoder(zs, gt_target)
        return out_cls, targets, sout, means, log_vars, zs


class Encoder(nn.Module):
    CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self):
        nn.Module.__init__(self)

        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS

        # Block 1
        self.block1 = nn.Sequential(
            ME.MinkowskiConvolution(1, ch[0], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
        )

        self.block2 = nn.Sequential(
            ME.MinkowskiConvolution(ch[0], ch[1], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )

        self.block3 = nn.Sequential(
            ME.MinkowskiConvolution(ch[1], ch[2], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )

        self.block4 = nn.Sequential(
            ME.MinkowskiConvolution(ch[2], ch[3], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )

        self.block5 = nn.Sequential(
            ME.MinkowskiConvolution(ch[3], ch[4], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
        )

        self.block6 = nn.Sequential(
            ME.MinkowskiConvolution(ch[4], ch[5], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
        )

        self.block7 = nn.Sequential(
            ME.MinkowskiConvolution(ch[5], ch[6], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
        )

        self.global_pool = ME.MinkowskiGlobalPooling()

        self.linear_mean = ME.MinkowskiLinear(ch[6], ch[6], bias=True)
        self.linear_log_var = ME.MinkowskiLinear(ch[6], ch[6], bias=True)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, sinput):
        out = self.block1(sinput)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        # out = self.global_pool(out)
        mean = self.linear_mean(out)
        log_var = self.linear_log_var(out)
        return mean, log_var


class Decoder(nn.Module):
    CHANNELS = [1024, 512, 256, 128, 64, 32, 16]
    resolution = 128

    def __init__(self):
        nn.Module.__init__(self)

        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS

        # Block 1
        self.block1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[0], ch[0], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[0], ch[1], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )

        self.block1_cls = ME.MinkowskiConvolution(
            ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 2
        self.block2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[1], ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )

        self.block2_cls = ME.MinkowskiConvolution(
            ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 3
        self.block3 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[2], ch[3], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )

        self.block3_cls = ME.MinkowskiConvolution(
            ch[3], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 4
        self.block4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[3], ch[4], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
        )

        self.block4_cls = ME.MinkowskiConvolution(
            ch[4], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 5
        self.block5 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[4], ch[5], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
        )

        self.block5_cls = ME.MinkowskiConvolution(
            ch[5], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 6
        self.block6 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[5], ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
        )

        self.block6_cls = ME.MinkowskiConvolution(
            ch[6], 1, kernel_size=1, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(target_key, out.tensor_stride[0])
        kernel_map = cm.kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, z_glob, target_key):
        out_cls, targets = [], []

        z = ME.SparseTensor(
            features=z_glob.F,
            coordinates=z_glob.C,
            tensor_stride=self.resolution,
            coordinate_manager=z_glob.coordinate_manager,
        )

        # Block1
        out1 = self.block1(z)
        out1_cls = self.block1_cls(out1)
        target = self.get_target(out1, target_key)
        targets.append(target)
        out_cls.append(out1_cls)
        keep1 = (out1_cls.F > 0).squeeze()

        # If training, force target shape generation, use net.eval() to disable
        if self.training:
            keep1 += target

        # Remove voxels 32
        out1 = self.pruning(out1, keep1)

        # Block 2
        out2 = self.block2(out1)
        out2_cls = self.block2_cls(out2)
        target = self.get_target(out2, target_key)
        targets.append(target)
        out_cls.append(out2_cls)
        keep2 = (out2_cls.F > 0).squeeze()

        if self.training:
            keep2 += target

        # Remove voxels 16
        out2 = self.pruning(out2, keep2)

        # Block 3
        out3 = self.block3(out2)
        out3_cls = self.block3_cls(out3)
        target = self.get_target(out3, target_key)
        targets.append(target)
        out_cls.append(out3_cls)
        keep3 = (out3_cls.F > 0).squeeze()

        if self.training:
            keep3 += target

        # Remove voxels 8
        out3 = self.pruning(out3, keep3)

        # Block 4
        out4 = self.block4(out3)
        out4_cls = self.block4_cls(out4)
        target = self.get_target(out4, target_key)
        targets.append(target)
        out_cls.append(out4_cls)
        keep4 = (out4_cls.F > 0).squeeze()

        if self.training:
            keep4 += target

        # Remove voxels 4
        out4 = self.pruning(out4, keep4)

        # Block 5
        out5 = self.block5(out4)
        out5_cls = self.block5_cls(out5)
        target = self.get_target(out5, target_key)
        targets.append(target)
        out_cls.append(out5_cls)
        keep5 = (out5_cls.F > 0).squeeze()

        if self.training:
            keep5 += target

        # Remove voxels 2
        out5 = self.pruning(out5, keep5)

        # Block 5
        out6 = self.block6(out5)
        out6_cls = self.block6_cls(out6)
        target = self.get_target(out6, target_key)
        targets.append(target)
        out_cls.append(out6_cls)
        keep6 = (out6_cls.F > 0).squeeze()

        # Last layer does not require keep
        # if self.training:
        #   keep6 += target

        # Remove voxels 1
        if keep6.sum() > 0:
            out6 = self.pruning(out6, keep6)

        return out_cls, targets, out6

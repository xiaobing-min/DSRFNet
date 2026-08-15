import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MobileNetV2
import numbers
from einops import rearrange
class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        context = self.conv_context(context)
        x = x.mul(context)
        x_out = self.conv2(x)

        return x_out, mask

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# LayerNorm组件
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# 频域自注意力模块
class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super(FSAS, self).__init__()
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')
        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        # 处理不能整除patch_size的情况
        b, c, h, w = q.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h), mode='reflect')
            k = F.pad(k, (0, pad_w, 0, pad_h), mode='reflect')
            v = F.pad(v, (0, pad_w, 0, pad_h), mode='reflect')

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
                            patch1=self.patch_size, patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
                            patch1=self.patch_size, patch2=self.patch_size)

        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))

        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
                        patch1=self.patch_size, patch2=self.patch_size)

        # 裁剪回原始尺寸
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
            v = v[:, :, :h, :w]

        out = self.norm(out)
        output = v * out
        output = self.project_out(output)
        return output


# 语义流场模块
class Semantic_flow(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Semantic_flow, self).__init__()
        self.down_h = nn.Conv2d(inchannel, outchannel, 1, bias=False)
        self.down_l = nn.Conv2d(outchannel, outchannel, 1, bias=False)
        self.flow_make = nn.Conv2d(outchannel * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, h_feature, low_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)

        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)

        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    @staticmethod
    def flow_warp(inputs, flow, size):
        out_h, out_w = size
        n, c, h, w = inputs.size()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(inputs).to(inputs.device)

        w_coords = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_coords = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_coords.unsqueeze(2), w_coords.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(inputs, grid, align_corners=False)
        return output


# 频域-流场融合上采样模块
class FrequencyFlowUpsampler(nn.Module):
    def __init__(self, high_dim, low_dim, out_dim):
        super(FrequencyFlowUpsampler, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.out_dim = out_dim

        # 语义流场对齐
        self.semantic_flow = Semantic_flow(high_dim, low_dim)

        # 频域注意力增强
        self.fsas = FSAS(low_dim)

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(low_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # 输出细化
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_feature, low_feature):
        """
        high_feature: 高层特征 (低分辨率)
        low_feature: 低层特征 (高分辨率)
        """
        # 1. 使用语义流场对齐高层特征
        aligned_high = self.semantic_flow(high_feature, low_feature)

        # 2. 频域注意力增强低层特征
        enhanced_low = self.fsas(low_feature)

        # 3. 特征融合
        fused_feature = torch.cat([aligned_high, enhanced_low], dim=1)
        fused_feature = self.fusion_conv(fused_feature)

        # 4. 输出细化
        output = self.refine_conv(fused_feature)

        return output
class Decoder(nn.Module):
    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d

        # 监督注意力模块
        self.sam_p5 = SupervisedAttentionModule(self.mid_d)
        self.sam_p4 = SupervisedAttentionModule(self.mid_d)
        self.sam_p3 = SupervisedAttentionModule(self.mid_d)

        # 频域-流场融合上采样模块
        self.ff_upsample_p4 = FrequencyFlowUpsampler(self.mid_d, self.mid_d, self.mid_d)
        self.ff_upsample_p3 = FrequencyFlowUpsampler(self.mid_d, self.mid_d, self.mid_d)
        self.ff_upsample_p2 = FrequencyFlowUpsampler(self.mid_d, self.mid_d, self.mid_d)
        # self.ff_upsample_p4 = FlowUpsampler(self.mid_d, self.mid_d, self.mid_d)
        # self.ff_upsample_p3 = FlowUpsampler(self.mid_d, self.mid_d, self.mid_d)
        # self.ff_upsample_p2 = FlowUpsampler(self.mid_d, self.mid_d, self.mid_d)

        # 最终分类层
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5):
        # 高层特征处理
        p5, mask_p5 = self.sam_p5(d5)

        # P4层: 使用频域-流场融合上采样
        p4 = self.ff_upsample_p4(p5, d4)
        p4, mask_p4 = self.sam_p4(p4)

        # P3层: 使用频域-流场融合上采样
        p3 = self.ff_upsample_p3(p4, d3)
        p3, mask_p3 = self.sam_p3(p3)

        # P2层: 使用频域-流场融合上采样
        p2 = self.ff_upsample_p2(p3, d2)
        mask_p2 = self.cls(p2)

        return p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5


class CrossModalFeatureRectification(nn.Module):
    def __init__(self, channels, reduction=4):
        super(CrossModalFeatureRectification, self).__init__()

        # 1. 动态卷积核生成器 (Dynamic Kernel Generator)
        # 这个部分根据T1的特征，为T2生成一个定制的3x3卷积核
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局信息编码
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            # 输出通道数为 channels * 9，因为我们要生成 channels 个 3x3 的核
            nn.Conv2d(channels // reduction, channels * 9, 1, bias=True)
        )

        # 2. 动态卷积使用的偏置 (bias) 生成器
        self.bias_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True)
        )

        self.group_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

    def forward(self, f_t1, f_t2):
        """
        f_t1: "干净"的T1分支特征 (guide feature)
        f_t2: "有雾"的T2分支特征 (feature to be rectified)
        """
        b, c, h, w = f_t2.size()

        # --- Part 1: 特征统计对齐 (AdaIN-like) ---
        # 计算T1和T2特征的均值和标准差
        mu_t1 = torch.mean(f_t1.view(b, c, -1), dim=2).view(b, c, 1, 1)
        std_t1 = torch.std(f_t1.view(b, c, -1), dim=2).view(b, c, 1, 1) + 1e-6

        mu_t2 = torch.mean(f_t2.view(b, c, -1), dim=2).view(b, c, 1, 1)
        std_t2 = torch.std(f_t2.view(b, c, -1), dim=2).view(b, c, 1, 1) + 1e-6

        # 将T2的统计分布对齐到T1
        f_t2_normalized = (f_t2 - mu_t2) / std_t2
        f_t2_aligned = f_t2_normalized * std_t1 + mu_t1

        # --- Part 2: 动态内容滤波 ---
        # 根据T1特征生成动态卷积核权重和偏置
        dynamic_kernel = self.kernel_generator(f_t1).view(b, c, 9, 1, 1)  # [B, C, 9, 1, 1]
        dynamic_bias = self.bias_generator(f_t1).view(b, c)  # [B, C]

        # 将动态核权重赋给分组卷积
        f_t2_unfold = F.unfold(f_t2_aligned, kernel_size=3, padding=1)  # [B, C*9, H*W]
        f_t2_unfold = f_t2_unfold.view(b, c, 9, h * w)  # [B, C, 9, H*W]

        # [B, C, 9, 1, 1] * [B, C, 9, H*W] -> [B, C, H*W]
        rectified_feature = (dynamic_kernel.squeeze(-1) * f_t2_unfold).sum(dim=2)
        rectified_feature = rectified_feature.view(b, c, h, w) + dynamic_bias.view(b, c, 1, 1)

        # --- Part 3: 残差融合 ---
        # 将矫正后的特征与原始T2特征相加，稳定训练
        f_t2_final = f_t2 + rectified_feature

        return f_t1, f_t2_final
class BaseNet(nn.Module):
    def __init__(self, backbone_name='mobilenetv2', fpn_name='fpn', fpn_channels=None,
                 deform_groups=4, gamma_mode='SE', beta_mode='gatedconv',
                 num_heads=1, num_points=8, kernel_layers=1, dropout_rate=0.1, init_type='kaiming_normal'):
        super(BaseNet, self).__init__()
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)
        channels = [16, 24, 32, 96, 320]
        #channles = [64, 64, 64, 64, 64]
        self.en_d = 32
        self.mid_d = self.en_d * 2
        self.decoder = Decoder(self.en_d * 2)
        # self.decoder = Decoder_baseline(self.en_d * 2)
        self.cfrm2 = CrossModalFeatureRectification(channels[1])
        self.cfrm3 = CrossModalFeatureRectification(channels[2])
        self.cfrm4 = CrossModalFeatureRectification(channels[3])
        self.cfrm5 = CrossModalFeatureRectification(channels[4])

        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(channels[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(channels[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale2_c4 = nn.Sequential(
            nn.Conv2d(channels[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale2_c5 = nn.Sequential(
            nn.Conv2d(channels[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)

        x1_2, x2_2 = self.cfrm2(x1_2, x2_2)
        x1_3, x2_3 = self.cfrm3(x1_3, x2_3)
        x1_4, x2_4 = self.cfrm4(x1_4, x2_4)
        x1_5, x2_5 = self.cfrm5(x1_5, x2_5)

        c2 = self.conv_scale2_c2(torch.abs(x1_2-x2_2))
        c3 = self.conv_scale2_c3(torch.abs(x1_3-x2_3))
        c4 = self.conv_scale2_c4(torch.abs(x1_4-x2_4))
        c5 = self.conv_scale2_c5(torch.abs(x1_5-x2_5))


        p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(c2, c3, c4, c5)

        mask_p2 = F.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear')
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, scale_factor=(8, 8), mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear')
        mask_p4 = torch.sigmoid(mask_p4)
        mask_p5 = F.interpolate(mask_p5, scale_factor=(32, 32), mode='bilinear')
        mask_p5 = torch.sigmoid(mask_p5)

        return mask_p2, mask_p3, mask_p4, mask_p5



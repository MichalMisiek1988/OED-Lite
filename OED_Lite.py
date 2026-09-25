import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class ECA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        c = max(dim, 2)
        k = int(abs(torch.log2(torch.tensor(float(c))) + 2))
        if k % 2 == 0:
            k += 1
        k = max(3, k)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x):
        y = self.pool(x).flatten(2).transpose(1, 2)
        y = self.conv(y)
        y = torch.sigmoid(y).transpose(1, 2).unsqueeze(-1)
        return x * y


class TFBlockECA(nn.Module):
    def __init__(self, c):
        super().__init__()
        hid = max(4, int(c * 0.25))
        self.attn = ECA(c)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hid, 1),
            nn.GELU(),
            nn.Conv2d(hid, c, 1),
        )

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = DSConv(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)


class DeepSupervisionBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super().__init__()
        self.up = nn.Upsample(scale_factor=upscale_factor, mode="bilinear")
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)

    def forward(self, x):
        return self.up(self.conv(x))


class UpsampleSkip(nn.Module):
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear")
        return torch.cat([x, skip], 1)


class ConvStack(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnAct(in_channels, out_channels),
            ConvBnAct(out_channels, out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.expansion_conv = (
            ConvBnAct(in_channels, out_channels, 1, 1, 0)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.conv = nn.Sequential(
            ConvBnAct(out_channels, out_channels),
            ConvBnAct(out_channels, out_channels),
        )

    def forward(self, x):
        x = self.expansion_conv(x)
        return x + self.conv(x)


class ResidualConvStack(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class OED_Lite(nn.Module):
    def __init__(self, backbone_name="mobilevitv2_050", pretrained=True):
        super().__init__()

        self.filters = [16, 32, 64, 128, 256]

        self.inp_conv_stack = ConvStack(3, 16)

        tmp = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )

        reds = list(map(int, tmp.feature_info.reduction()))
        del tmp

        need = [4, 8, 16, 32]

        if any(r not in reds for r in need):
            raise RuntimeError(
                f"Backbone '{backbone_name}' does not provide all required strides {need}. "
                f"Available strides: {reds}"
            )

        out_indices = tuple(reds.index(r) for r in need)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        c4, c8, c16, c32 = map(int, self.backbone.feature_info.channels())

        self.adapt_r1 = nn.Conv2d(16, 24, 1, bias=False)
        self.adapt_r2 = nn.Conv2d(c4, 32, 1, bias=False)
        self.adapt_r3 = nn.Conv2d(c8, 48, 1, bias=False)
        self.adapt_r4 = nn.Conv2d(c16, 136, 1, bias=False)
        self.adapt_r6 = nn.Conv2d(c32, 1536, 1, bias=False)

        self.bot_tf = TFBlockECA(1536)

        self.level5_res_conv_stack = ResidualConvStack(136 + 1536, 256)
        self.level4_res_conv_stack = ResidualConvStack(48 + 256, 128)
        self.level3_res_conv_stack = ResidualConvStack(32 + 128, 64)
        self.level2_res_conv_stack = ResidualConvStack(24 + 64, 32)
        self.level1_conv_stack = ConvStack(16 + 32, 16)

        self.upsample_skip = UpsampleSkip()

        self.level5_dsv = DeepSupervisionBlock(256, 16)
        self.level4_dsv = DeepSupervisionBlock(128, 8)
        self.level3_dsv = DeepSupervisionBlock(64, 4)
        self.level2_dsv = DeepSupervisionBlock(32, 2)

        self.final_out = nn.Conv2d(16, 1, 3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def _dyn_weight(self, d2, d3, d4, d5):
        outs = [d2, d3, d4, d5]
        target_size = outs[0].shape[-2:]

        outs = [
            t if t.shape[-2:] == target_size
            else F.interpolate(
                t,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            for t in outs
        ]

        scores = torch.stack(
            [self.pool(t).flatten(1).squeeze(1) for t in outs],
            dim=1,
        )

        weights = torch.softmax(scores, dim=1)

        return sum(
            t * weights[:, i].view(-1, 1, 1, 1)
            for i, t in enumerate(outs)
        )

    def forward(self, x):
        inp = self.inp_conv_stack(x)

        f4, f8, f16, f32 = self.backbone(x)

        r1 = self.adapt_r1(
            F.interpolate(
                inp,
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
            )
        )
        r2 = self.adapt_r2(f4)
        r3 = self.adapt_r3(f8)
        r4 = self.adapt_r4(f16)
        r6 = self.adapt_r6(f32)

        x = self.bot_tf(r6)

        x = self.level5_res_conv_stack(self.upsample_skip(x, r4))
        d5 = self.level5_dsv(x)

        x = self.level4_res_conv_stack(self.upsample_skip(x, r3))
        d4 = self.level4_dsv(x)

        x = self.level3_res_conv_stack(self.upsample_skip(x, r2))
        d3 = self.level3_dsv(x)

        x = self.level2_res_conv_stack(self.upsample_skip(x, r1))
        d2 = self.level2_dsv(x)

        x = self.level1_conv_stack(self.upsample_skip(x, inp))
        final_out = self.final_out(x)

        size = final_out.shape[-2:]

        d2 = F.interpolate(d2, size=size, mode="bilinear", align_corners=False)
        d3 = F.interpolate(d3, size=size, mode="bilinear", align_corners=False)
        d4 = F.interpolate(d4, size=size, mode="bilinear", align_corners=False)
        d5 = F.interpolate(d5, size=size, mode="bilinear", align_corners=False)

        logits = 0.5 * (final_out + self._dyn_weight(d2, d3, d4, d5))

        return d2, d3, d4, d5, logits

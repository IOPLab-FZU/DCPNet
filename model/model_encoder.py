import torch
from torch import nn
import torchvision.models as models
from einops import rearrange
import math
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network
        if self.network=='alexnet': #256,7,7
            cnn = models.alexnet(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg11': #512,1/32H,1/32W
            cnn = models.vgg11(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg16': #512,1/32H,1/32W
            cnn = models.vgg16(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg19':#512,1/32H,1/32W
            cnn = models.vgg19(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='inception': #2048,6,6
            cnn = models.inception_v3(pretrained=True, aux_logits=False)  
            modules = list(cnn.children())[:-3]
        elif self.network=='resnet18': #512,1/32H,1/32W
            cnn = models.resnet18(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet34': #512,1/32H,1/32W
            cnn = models.resnet34(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet50': #2048,1/32H,1/32W
            cnn = models.resnet50(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet101':  #2048,1/32H,1/32W
            cnn = models.resnet101(pretrained=True)  
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet152': #512,1/32H,1/32W
            cnn = models.resnet152(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext50_32x4d': #2048,1/32H,1/32W
            cnn = models.resnext50_32x4d(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext101_32x8d':#2048,1/256H,1/256W
            cnn = models.resnext101_32x8d(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='densenet121': #no AdaptiveAvgPool2d #1024,1/32H,1/32W
            cnn = models.densenet121(pretrained=True) 
            modules = list(cnn.children())[:-1] 
        elif self.network=='densenet169': #1664,1/32H,1/32W
            cnn = models.densenet169(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='densenet201': #1920,1/32H,1/32W
            cnn = models.densenet201(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='regnet_x_400mf': #400,1/32H,1/32W
            cnn = models.regnet_x_400mf(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='regnet_x_8gf': #1920,1/32H,1/32W
            cnn = models.regnet_x_8gf(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='regnet_x_16gf': #2048,1/32H,1/32W
            cnn = models.regnet_x_16gf(pretrained=True) 
            modules = list(cnn.children())[:-2]

        self.cnn = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, imageA, imageB):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        feat1 = self.cnn(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
        feat2 = self.cnn(imageB)

        return feat1, feat2

    def fine_tune(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.cnn.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.cnn.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(4, d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(4, d_model))

        self.x_proj = nn.Linear(d_model, (self.dt_rank + d_state * 2) * 4)
        self.dt_projs = nn.ModuleList([nn.Linear(self.dt_rank, d_model) for _ in range(4)])

    def forward(self, x):
        B_batch, E, H, W = x.shape
        L = H * W

        x_hw = x.view(B_batch, E, L)
        x_wh = x.transpose(2, 3).contiguous().view(B_batch, E, L)
        x_hw_flip = torch.flip(x, dims=[2, 3]).view(B_batch, E, L)
        x_wh_flip = torch.flip(x.transpose(2, 3).contiguous(), dims=[2, 3]).view(B_batch, E, L)

        xs = [x_hw, x_wh, x_hw_flip, x_wh_flip]

        x_for_proj = x_hw.transpose(1, 2)
        proj_out = self.x_proj(x_for_proj)
        proj_out = proj_out.view(B_batch, L, 4, -1).permute(2, 0, 1, 3)

        ys = []
        for i in range(4):
            dt_b_c = proj_out[i]
            dt_raw = dt_b_c[:, :, :self.dt_rank]
            B_raw = dt_b_c[:, :, self.dt_rank : self.dt_rank + self.d_state]
            C_raw = dt_b_c[:, :, self.dt_rank + self.d_state :]

            dt = F.softplus(self.dt_projs[i](dt_raw)).transpose(1, 2)
            B_val = B_raw.transpose(1, 2)
            C_val = C_raw.transpose(1, 2)

            A_val = -torch.exp(self.A_log[i])
            D_val = self.D[i]

            y = selective_scan_fn(
                xs[i].contiguous(),
                dt.contiguous(),
                A_val.contiguous(),
                B_val.contiguous(),
                C_val.contiguous(),
                D_val.contiguous(),
                z=None,
                delta_bias=None,
                delta_softplus=False,
                return_last_state=False
            )
            
            ys.append(y)

        y_hw = ys[0].view(B_batch, E, H, W)
        y_wh = ys[1].view(B_batch, E, W, H).transpose(2, 3).contiguous()
        y_hw_flip = torch.flip(ys[2].view(B_batch, E, H, W), dims=[2, 3])
        y_wh_flip = torch.flip(ys[3].view(B_batch, E, W, H).transpose(2, 3).contiguous(), dims=[2, 3])

        return y_hw + y_wh + y_hw_flip + y_wh_flip

class VSSBlock(nn.Module):
    """
    Visual State Space Block
    """
    def __init__(self, in_channels, expand_ratio=2, d_state=16):
        super().__init__()
        self.in_channels = in_channels
        self.d_inner = int(in_channels * expand_ratio)

        self.ln_1 = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, self.d_inner * 2)
        
        self.dwconv = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=3,
            padding=1, groups=self.d_inner, bias=False
        )
        self.act = nn.SiLU()
        self.ss2d = SS2D(d_model=self.d_inner, d_state=d_state)
        self.ln_2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, in_channels)

    def forward(self, x):
        shortcut = x
        x_norm = self.ln_1(x.permute(0, 2, 3, 1))
        
        x_proj = self.in_proj(x_norm)
        x1, x2 = x_proj.chunk(2, dim=-1)

        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = self.dwconv(x1)
        x1 = self.act(x1)
        x1 = self.ss2d(x1)
        x1 = self.ln_2(x1.permute(0, 2, 3, 1))

        x2 = self.act(x2)

        out = x1 * x2
        out = self.out_proj(out).permute(0, 3, 1, 2).contiguous()

        return out + shortcut

class CrossAttention(nn.Module):

    def __init__(self, channels, reduction=8, topk=32):
        super().__init__()
        self.channels = channels
        self.inter_channels = max(channels // reduction, 32)
        self.topk = topk

        self.q_proj = nn.Conv2d(channels, self.inter_channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, self.inter_channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, self.inter_channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(self.inter_channels, channels, kernel_size=1, bias=False)

    def _build_sparse_attn(self, q, k):
        attn = torch.matmul(q, k) / math.sqrt(q.size(-1))
        n = attn.size(-1)
        k_top = min(self.topk, n)
        topv, topi = torch.topk(attn, k=k_top, dim=-1)
        mask = torch.full_like(attn, float('-inf'))
        mask.scatter_(-1, topi, topv)
        attn = F.softmax(mask, dim=-1)
        return attn

    def forward(self, x1, x2):
        b, _, h, w = x1.shape

        q1 = self.q_proj(x1).flatten(2).transpose(1, 2).contiguous()
        k1 = self.k_proj(x1).flatten(2).contiguous()
        v1 = self.v_proj(x1).flatten(2).transpose(1, 2).contiguous()

        q2 = self.q_proj(x2).flatten(2).transpose(1, 2).contiguous()
        k2 = self.k_proj(x2).flatten(2).contiguous()
        v2 = self.v_proj(x2).flatten(2).transpose(1, 2).contiguous()

        attn12 = self._build_sparse_attn(q1, k2)
        attn21 = self._build_sparse_attn(q2, k1)

        ctx12 = torch.matmul(attn12, v2).transpose(1, 2).contiguous().view(b, self.inter_channels, h, w)
        ctx21 = torch.matmul(attn21, v1).transpose(1, 2).contiguous().view(b, self.inter_channels, h, w)

        return self.out_proj(ctx12), self.out_proj(ctx21)


class PromptFusion(nn.Module):
    """
    Fuse directional difference, common activation, and sparse cross context into a directional prompt.
    """

    def __init__(self, channels, hidden_ratio=0.5):
        super().__init__()
        hidden = max(int(channels * hidden_ratio), 64)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, dir_diff, prod, cross):
        return self.fuse(torch.cat([dir_diff, prod, cross], dim=1))


class SymmetricAnchorAlignmentLoss(nn.Module):
    """
    Keep contrastive/triplet learning, but use a fused symmetric anchor instead of
    directly pulling the two directional branches toward each other.
    """

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def _triplet(self, anchor, positive, negatives):
        pos = F.cosine_similarity(anchor, positive, dim=-1)
        sim = torch.matmul(anchor, negatives.t())
        neg_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        hardest_neg = sim.masked_fill(~neg_mask, float('-inf')).max(dim=1).values
        return F.relu(self.margin + hardest_neg - pos).mean()

    def forward(self, forward_feat, backward_feat):
        if forward_feat.size(0) <= 1:
            return forward_feat.new_zeros(())

        z_f = F.normalize(forward_feat, dim=-1)
        z_b = F.normalize(backward_feat, dim=-1)
        z_s = F.normalize(0.5 * (z_f + z_b), dim=-1)

        loss_f = self._triplet(z_f, z_s, z_s)
        loss_b = self._triplet(z_b, z_s, z_s)
        return 0.5 * (loss_f + loss_b)


class PromptGuidedVSSBlock(nn.Module):
    """Cross attention acts as a shallow change prompt, while SS2D remains the main modeling path."""

    def __init__(self, in_channels, expand_ratio=2, d_state=16, topk=32, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.d_inner = int(in_channels * expand_ratio)

        self.ln_1 = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, self.d_inner * 2)
        self.dwconv = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner, bias=False
        )
        self.act = nn.SiLU()

        self.cross_attention = CrossAttention(self.d_inner, reduction=reduction, topk=topk)
        self.forward_prompt_fuse = PromptFusion(self.d_inner, hidden_ratio=0.5)
        self.backward_prompt_fuse = PromptFusion(self.d_inner, hidden_ratio=0.5)
        self.beta1 = nn.Parameter(torch.tensor(0.1))
        self.beta2 = nn.Parameter(torch.tensor(0.1))

        self.ss2d = SS2D(d_model=self.d_inner, d_state=d_state)
        self.saa_loss = SymmetricAnchorAlignmentLoss(margin=0.2)
        self.ln_2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, in_channels)

    def _project_local(self, x):
        x_norm = self.ln_1(x.permute(0, 2, 3, 1))
        x_proj = self.in_proj(x_norm)
        x_main, x_gate = x_proj.chunk(2, dim=-1)
        x_main = x_main.permute(0, 3, 1, 2).contiguous()
        x_local = self.act(self.dwconv(x_main))
        return x_local, x_gate

    def _finish(self, x_main, x_gate, shortcut):
        x_main = self.ln_2(x_main.permute(0, 2, 3, 1))
        x_gate = self.act(x_gate)
        out = x_main * x_gate
        out = self.out_proj(out).permute(0, 3, 1, 2).contiguous()
        return out + shortcut

    def forward(self, x1, x2, return_aux=False):
        shortcut1, shortcut2 = x1, x2

        x_local1, x_gate1 = self._project_local(x1)
        x_local2, x_gate2 = self._project_local(x2)

        forward_diff = F.relu(x_local2 - x_local1)
        backward_diff = F.relu(x_local1 - x_local2)
        prod = x_local1 * x_local2
        cross12, cross21 = self.cross_attention(x_local1, x_local2)

        forward_prompt = self.forward_prompt_fuse(forward_diff, prod, cross12)
        backward_prompt = self.backward_prompt_fuse(backward_diff, prod, cross21)

        x_local1 = x_local1 + self.beta1 * backward_prompt
        x_local2 = x_local2 + self.beta2 * forward_prompt

        x_ssm1 = self.ss2d(x_local1)
        x_ssm2 = self.ss2d(x_local2)

        out1 = self._finish(x_ssm1, x_gate1, shortcut1)
        out2 = self._finish(x_ssm2, x_gate2, shortcut2)

        if return_aux:
            prompt_vec1 = F.adaptive_avg_pool2d(forward_prompt, 1).flatten(1)
            prompt_vec2 = F.adaptive_avg_pool2d(backward_prompt, 1).flatten(1)
            aux = {
                'saa_loss': self.saa_loss(prompt_vec1, prompt_vec2),
                'prompt_vec1': prompt_vec1,
                'prompt_vec2': prompt_vec2,
            }
            return out1, out2, aux

        return out1, out2


class DCPNetEncoder(nn.Module):
    def __init__(
        self,
        n_layers,
        feature_size,
        heads,
        hidden_dim,
        semantic_dim=512,
        dropout=0.1,
        num_prompt_layers=2,
        topk=32,
    ):
        super().__init__()
        _, _, channels = feature_size
        channels = 2048

        self.sem = nn.Sequential(
            nn.Conv2d(channels, semantic_dim, kernel_size=1),
            nn.BatchNorm2d(semantic_dim),
            nn.ReLU(),
            nn.Conv2d(semantic_dim, semantic_dim, kernel_size=1),
            nn.BatchNorm2d(semantic_dim),
            nn.ReLU(),
            nn.Conv2d(semantic_dim, semantic_dim, kernel_size=3, padding=1, groups=semantic_dim),
        )

        self.num_prompt_layers = min(num_prompt_layers, n_layers)
        self.prompt_layers = nn.ModuleList([
            PromptGuidedVSSBlock(in_channels=semantic_dim, expand_ratio=2, d_state=16, topk=topk, reduction=8)
            for _ in range(self.num_prompt_layers)
        ])
        self.plain_layers = nn.ModuleList([
            VSSBlock(in_channels=semantic_dim, expand_ratio=2, d_state=16)
            for _ in range(n_layers - self.num_prompt_layers)
        ])

    def forward(self, img1, img2, return_aux=False):
        img1 = self.sem(img1)
        img2 = self.sem(img2)

        aux_losses = []
        for layer in self.prompt_layers:
            if return_aux:
                img1, img2, aux = layer(img1, img2, return_aux=True)
                aux_losses.append(aux['saa_loss'])
            else:
                img1, img2 = layer(img1, img2)

        for layer in self.plain_layers:
            img1 = layer(img1)
            img2 = layer(img2)

        if return_aux:
            saa_loss = torch.stack(aux_losses).mean() if aux_losses else img1.new_zeros(())
            return img1, img2, {'saa_loss': saa_loss}
        return img1, img2

import torch
import torch.nn as nn

from .swin import SwinTransformer
from .ssast_models import SSASTModel
from einops import rearrange
from torchlibrosa.augmentation import SpecAugmentation

class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class MSQAT(nn.Module):
    def __init__(self, embed_dim=768, num_outputs=1,  drop=0.1, input_fdim=128, input_tdim=1024,
                 depths=[2, 2, 6, 2], window_size=4, dim_mlp=768, num_heads=[4, 8, 16, 32], fstride=16, tstride=16,
                 fshape=16, tshape=16, model_size='base', load_pretrained_mdl_path='./test_mdl.pth',
                 num_tab=2, scale=0.8, **kwargs):
        super().__init__()

        self.ast = SSASTModel(label_dim=1,
                              fshape=fshape, tshape=tshape, fstride=fstride, tstride=tstride,
                              input_fdim=input_fdim, input_tdim=input_tdim, model_size=model_size,
                              pretrain_stage=False, load_pretrained_mdl_path=load_pretrained_mdl_path)

        self.f_dim = input_fdim // fshape
        self.t_dim = input_tdim // tshape

        self.spec_augmenter = SpecAugmentation(time_drop_width=128, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2) # 2 2

        self.conv1 = nn.Conv2d(embed_dim * 3, embed_dim, 1, 1, 0)
        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.f_dim * self.t_dim)
            self.tablock1.append(tab)
        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.f_dim * self.t_dim)
            self.tablock2.append(tab)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=(self.f_dim, self.t_dim),
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )
        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=(self.f_dim, self.t_dim),
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim, num_outputs),
            nn.Sigmoid()
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(
            embed_dim), nn.Linear(embed_dim, num_outputs))

    def forward(self, x):
        x = x.unsqueeze(1)
        est_x = self.spec_augmenter(x)
        x = x.squeeze(1)
        est_x = est_x.squeeze(1)
        x = self.ast(x)  # 1 512 768
        est_x = self.ast(est_x)
        # stage 1
        res_x = torch.abs(x - est_x)
        x = torch.concat((x,est_x,res_x),dim=2)
        x = rearrange(x, 'b (h w) c -> b c (h w)',
                      h=self.f_dim, w=self.t_dim)  # 1 768 512
        
        for tab in self.tablock1:
            x = tab(x)
        
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.f_dim,
                      w=self.t_dim)  # 1 768 28 28
        x = self.conv1(x)
        x = self.swintransformer1(x)

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.f_dim, w=self.t_dim)
        # x = torch.mean(x, dim=1)
        # score = self.mlp_head(x)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        score = score.unsqueeze(1)
        return score


if __name__ == '__main__':
    input_tdim = 1024
    ast_mdl = MSQAT(input_tdim=input_tdim, fstride=128,
                     fshape=128, tstride=2, tshape=2).cuda()
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([1, input_tdim, 128]).cuda()
    test_output = ast_mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)

    # input_tdim = 256
    # ast_mdl = ASTModel(input_tdim=input_tdim,label_dim=50, audioset_pretrain=True)
    # # input a batch of 10 spectrogram, each with 512 time frames and 128 frequency bins
    # test_input = torch.rand([10, input_tdim, 128])
    # test_output = ast_mdl(test_input)
    # # output should be in shape [10, 50], i.e., 10 samples, each with prediction of 50 classes.
    # print(test_output.shape)

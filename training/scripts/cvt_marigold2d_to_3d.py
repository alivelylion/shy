import sys
import os.path as osp
import torch


sys.path.append(osp.abspath(osp.dirname(osp.dirname(__file__))))

from utils.torch_utils import seed_everything
from utils.io_utils import json2dict
from utils.torch_utils import init_model_from_pretrained, _get_model_file, load_state_dict
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from diffusers.models import UNet2DConditionModel


seed_everything(0)

src_dir = 'workspace/training_output/finetune_marigolddepth_iter1'
subfolder = None
save_dir = 'workspace/ckpt/metricdepth3d_tmp'
configp = 'common/assets/marigold3d.json'
weights_name = 'checkpoint-20000.safetensors'

device = 'cuda'
dtype = torch.bfloat16

seq_len = 2
batch_size = 1

size = 512

model_type = 'sd1.5'

if model_type != 'sd1.5':
    added_cond_kwargs = {
        'text_embeds': torch.randn((batch_size, seq_len, 1280), device=device, dtype=dtype).reshape(batch_size * seq_len, -1),
        'time_ids': torch.tensor([size, size, 0, 0, size, size])[None, None].repeat(batch_size, seq_len, 1).reshape(batch_size * seq_len, -1).to(device=device, dtype=dtype)
    }
    encoder_hidden_states = torch.randn((batch_size, seq_len, 77, 2048), device=device, dtype=dtype)
else:
    added_cond_kwargs = None
    encoder_hidden_states = torch.randn((batch_size, seq_len, 77, 1024), device=device, dtype=dtype)

encoder_hidden_states = encoder_hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, *encoder_hidden_states.shape[2:])

latent = torch.randn((batch_size, seq_len, 8, size // 8, size // 8), device=device, dtype=dtype)

timesteps = torch.LongTensor([256]).to(device=device)


model: UNetFrameConditionModel = init_model_from_pretrained(
    src_dir,
    UNetFrameConditionModel.from_config, 
    model_args=dict(config=json2dict(configp)),
    subfolder=subfolder,
    weights_name=weights_name
)


model.init_weights()
model.to(device=device, dtype=dtype)
model.save_pretrained(save_dir)


with torch.inference_mode():
    pred_layerdiff = model(
        latent,
        timesteps,
        encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False
    )[0]
model.to(device='cpu')
del model

# 2 320 64 64
# 2 1280
# 2 77 768

latent.reshape(batch_size * seq_len, *latent.shape[2:])

model: UNet2DConditionModel = init_model_from_pretrained(
    src_dir,
    UNet2DConditionModel.from_config, 
    model_args=dict(config=json2dict('common/assets/marigold.json')),
    subfolder=subfolder,
    weights_name=weights_name
)
model.to(device=device, dtype=dtype)

with torch.inference_mode():
    pred = model(
        latent.reshape(batch_size * seq_len, *latent.shape[2:]),
        timesteps.repeat_interleave(seq_len),
        encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False
    )[0].reshape(batch_size, seq_len, 4, size // 8, size // 8)

print(torch.sum(torch.abs(pred_layerdiff - pred)))


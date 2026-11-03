import numpy as np
import torch


class AvgMeter:

    def __init__(self) -> None:
        self.meter_dict = {}

    def add(self, log_dict):
        for k, v in log_dict.items():
            if k in self.meter_dict:
                vlist = self.meter_dict[k]
            else:
                vlist = []
                self.meter_dict[k] = vlist
            if isinstance(v, torch.Tensor) and v.ndim == 0:
                v = v.to(device='cpu', dtype=torch.float32).item()
                num_input = 1
            if not isinstance(v, (float, int, np.ScalarType)):
                num_input = len(v)
                if isinstance(v, list):
                    for value in v:
                        if isinstance(value, torch.Tensor):
                            value = value.detach().cpu().item()
                        vlist.append(value)
                else:
                    v = v.mean()
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().item()
                    vlist += [v] * num_input
            else:
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item()
                vlist.append(v)

    def compute(self):
        log_dict = {}
        for k, vlist in self.meter_dict.items():
            log_dict[k] = np.array(vlist).mean()
        return log_dict
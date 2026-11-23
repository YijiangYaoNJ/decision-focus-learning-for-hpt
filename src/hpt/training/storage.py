import json

import torch


def record(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def atomic_save(value, path):
    tmp = path.with_suffix(".part")
    torch.save(value, tmp)
    tmp.replace(path)

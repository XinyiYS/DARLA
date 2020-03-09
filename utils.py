import os
import torch

def save_checkpoint(model_class, model, filepath):
    checkpoint = create_checkpoint(model_class, model)
    torch.save(checkpoint, filepath)
    return

def create_checkpoint(model_class, model, optimizer = None):
    checkpoint = {'model_class': model_class,
          'state_dict': model.state_dict(),}
    if optimizer:
          checkpoint['optimizer'] : optimizer.state_dict()
    return checkpoint

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model_class']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

def create_dir(directory):
    try:
        os.mkdir(directory)
    except OSError:
        pass
    return
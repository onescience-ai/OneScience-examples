from backend.ml_models.tornado_predictor import TornadoSuperPredictor as Model
import torch
def load(device='cpu'):
    m = Model().to(device)
    m.load_state_dict(torch.load('pytorch_model.bin', map_location=device))
    m.eval(); return m
def apply_temperature(logits, T):
    return logits / max(T,1e-6)

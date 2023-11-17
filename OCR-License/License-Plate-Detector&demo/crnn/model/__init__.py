from .model import get_crnn

def get_model(config,converter):
    if config['model_type'] == "CRNN":
        return get_crnn(config,converter)
    else:
        raise NotImplemented()
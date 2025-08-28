import torch
import torch.nn as nn
import torch.nn.init as init
import timm


class MobileViTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilevit = timm.create_model(
            'mobilevit_xxs',
            pretrained=False,
            features_only=True,
            out_indices=[2]  # Output features from stage 2 (spatial size: H/8, W/8)
        )

    def forward(self, x):
        return self.mobilevit(x)


def load_pretrained_weights(model, weight_path):
    state_dict = torch.load(weight_path, map_location='cpu')
    new_state_dict = {}

    for key in state_dict:
        new_key = key.replace('stages.', 'stages_')
        new_state_dict[new_key] = state_dict[key]

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    # Only use up to Stage 3; remaining layers are ignored
    # print("Unexpected keys:", unexpected_keys)  # Optional debug output for unmatched state dict keys

    return model, new_state_dict


def init_missing_parameters(model, new_state_dict):
    model_state_dict = model.state_dict()
    missing_keys = [k for k in model_state_dict if k not in new_state_dict]
    print("Initializing missing keys:", missing_keys)

    for name, param in model.named_parameters():
        if name in missing_keys:
            if 'conv' in name and 'weight' in name:
                init.kaiming_normal_(param)
            elif 'bn' in name and 'weight' in name:
                init.ones_(param)
            elif 'bn' in name and 'bias' in name:
                init.zeros_(param)
            elif 'running_mean' in name or 'running_var' in name:
                param.data.zero_()


def get_mobilevit_model(weight_path="pretrained/mobilevit_xxs-ad385b40.pth"):

    model = MobileViTFeatureExtractor()

    # Load pretrained weights and obtain the new state dictionary
    model.mobilevit, new_state_dict = load_pretrained_weights(model.mobilevit, weight_path)

    # Initialize missing parameters (e.g., for newly added layers)
    init_missing_parameters(model.mobilevit, new_state_dict)

    #  Freeze early stages: stages_0 and stages_1
    for name, param in model.mobilevit.named_parameters():
        if 'stages_0' in name or 'stages_1' in name:
            param.requires_grad = False

    return model.mobilevit
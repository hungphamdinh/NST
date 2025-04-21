import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from glob import glob
import copy
import os
# -------------------------
# 1. Device & Transforms
# -------------------------
def get_device_and_size():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size   = 512 if torch.cuda.is_available() else 128
    return device, size

def get_transforms(image_size):
    loader = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    unloader = transforms.ToPILImage()
    return loader, unloader

# -------------------------
# 2. Image I/O Utilities
# -------------------------
def load_image(path, loader, device):
    """Load an image file into a preprocessed tensor."""
    img = Image.open(path).convert("RGB")
    return loader(img).unsqueeze(0).to(device)

def show_image(tensor, unloader):
    """Display a tensor as a PIL image."""
    img = tensor.cpu().clone().squeeze(0)
    unloader(img).show()

# -------------------------
# 3. VGG Feature Extractor
# -------------------------
class VGGFeatures(nn.Module):
    def __init__(self, cnn, style_layers, content_layers):
        super().__init__()
        self.style_layers   = style_layers
        self.content_layers = content_layers
        self.model = self._build(cnn)

    def _build(self, cnn):
        seq    = nn.Sequential()
        conv_i = 0
        for layer in cnn.children():
            if   isinstance(layer, nn.Conv2d):   conv_i += 1; name = f"conv_{conv_i}"
            elif isinstance(layer, nn.ReLU):     name = f"relu_{conv_i}"; layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):name = f"pool_{conv_i}"
            elif isinstance(layer, nn.BatchNorm2d): name = f"bn_{conv_i}"
            else: continue
            seq.add_module(name, layer)
        return seq

    def forward(self, x):
        style_feats, content_feats = {}, {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.style_layers:
                style_feats[name] = x
            if name in self.content_layers:
                content_feats[name] = x
        return style_feats, content_feats

# -------------------------
# 4. Gram & Loss Modules
# -------------------------
def gram_matrix(x):
    b, c, h, w = x.size()
    f = x.view(c, h*w)
    G = torch.mm(f, f.t())
    return G.div(c*h*w)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss   = 0.0

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_gram, weight=1.0):
        super().__init__()
        self.target = target_gram.detach()
        self.weight = weight
        self.loss   = 0.0

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = self.weight * nn.functional.mse_loss(G, self.target)
        return input

class TVLoss(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return tv_h + tv_w

# -------------------------
# 5. Compute Average Style Grams
# -------------------------
def compute_average_grams(style_dir, loader, feature_extractor, device):
    # recursively find .jpg and .png in style_dir and all its subfolders
    pattern = os.path.join(style_dir, "**", "*.[jp][pn]g")
    paths   = glob(pattern, recursive=True)
    if not paths:
        raise ValueError(f"No images found under {style_dir}")
    # Initialize accumulators from first image
    sample_feats, _ = feature_extractor(load_image(paths[0], loader, device))
    avg_grams = {
        layer: torch.zeros_like(gram_matrix(sample_feats[layer]))
        for layer in sample_feats
    }
    # Sum and average
    with torch.no_grad():
        for p in paths:
            img = load_image(p, loader, device)
            s_feats, _ = feature_extractor(img)
            for layer, feat in s_feats.items():
                avg_grams[layer] += gram_matrix(feat)
        count = len(paths)
        for layer in avg_grams:
            avg_grams[layer] /= count
    return avg_grams

# 6. Build Loss‐Augmented Model
# -------------------------
def build_style_content_model(cnn, avg_grams, content_img,
                              style_layers, content_layers,
                              style_weights, device):
    cnn_copy = copy.deepcopy(cnn)
    seq      = nn.Sequential().to(device)
    content_losses, style_losses = [], []
    conv_i = 0

    for layer in cnn_copy.children():
        if   isinstance(layer, nn.Conv2d):    conv_i += 1; name = f"conv_{conv_i}"
        elif isinstance(layer, nn.ReLU):      name = f"relu_{conv_i}"; layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d): name = f"pool_{conv_i}"
        elif isinstance(layer, nn.BatchNorm2d): name = f"bn_{conv_i}"
        else: continue
        seq.add_module(name, layer)

        if name in content_layers:
            target = seq(content_img).clone()
            cl = ContentLoss(target)
            seq.add_module(f"content_loss_{conv_i}", cl)
            content_losses.append(cl)

        if name in style_layers:
            target_gram = avg_grams[name]
            w = style_weights.get(name, 1.0)
            sl = StyleLoss(target_gram, weight=w)
            seq.add_module(f"style_loss_{conv_i}", sl)
            style_losses.append(sl)

    # Prune layers after last loss
    for i in reversed(range(len(seq))):
        if isinstance(seq[i], (ContentLoss, StyleLoss)):
            seq = seq[:i+1]
            break

    return seq, style_losses, content_losses, TVLoss()

# -------------------------
# 7. Optimization Loop
# -------------------------
def run_style_transfer(model, style_losses, content_losses, tv_loss_fn,
                       input_img, optimizer, tv_weight,
                       steps=300, log_interval=50):
    run = [0]
    while run[0] <= steps:
        def closure():
            input_img.data.clamp_(0,1)
            optimizer.zero_grad()
            model(input_img)
            s = sum(sl.loss for sl in style_losses)
            c = sum(cl.loss for cl in content_losses)
            tv = tv_weight * tv_loss_fn(input_img)
            loss = s + c + tv
            loss.backward()
            run[0] += 1
            if run[0] % log_interval == 0:
                print(f"Step {run[0]}: style={s:.2f}, content={c:.2f}, tv={tv:.2f}")
            return loss
        optimizer.step(closure)

    input_img.data.clamp_(0,1)
    return input_img

# -------------------------
# 8. Main Entry Point
# -------------------------
if __name__ == "__main__":
    device, size = get_device_and_size()
    loader, unloader = get_transforms(size)

    # Content image
    content_path = "./data/generation/6.jpg"
    content_img  = load_image(content_path, loader, device)

    style_layers   = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    content_layers = ['conv_4']

    # VGG feature extractor
    cnn_base     = models.vgg19(pretrained=True).features.to(device).eval()
    vgg_feats    = VGGFeatures(cnn_base, style_layers, content_layers).to(device)

    # Average style Grams
    avg_grams    = compute_average_grams("./data/ghibli_dataset",
                                         loader, vgg_feats, device)

    # Build model with losses
    style_weights = {L: 1e6/len(style_layers) for L in style_layers}
    tv_weight     = 1e-6
    model, sl, cl, tv_fn = build_style_content_model(
        cnn_base, avg_grams, content_img,
        style_layers, content_layers,
        style_weights, device
    )

    # Optimize
    input_img  = content_img.clone().requires_grad_(True)
    optimizer  = optim.LBFGS([input_img])
    output     = run_style_transfer(model, sl, cl, tv_fn,
                                    input_img, optimizer, tv_weight)

    # Save result
    result = output.cpu().clone().squeeze(0)
    unloader(result).save("ghibli_ensemble_stylized6.jpg")
    print("Done — output saved as ghibli_ensemble_stylized.jpg")

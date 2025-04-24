import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
from glob import glob
import logging
from tqdm import tqdm

# -------------------------
# 1. Utility functions
# -------------------------
def gram_matrix(x):
    b, c, h, w = x.size()
    f = x.view(b, c, h * w)
    G = torch.bmm(f, f.transpose(1, 2))
    return G.div(c * h * w)

# -------------------------
# 2. VGG feature extractor
# -------------------------
class VGGFeatures(nn.Module):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        for p in vgg.parameters(): p.requires_grad_(False)
        self.style_layers   = style_layers
        self.content_layers = content_layers

        self.layers = nn.ModuleList()
        self.names  = []
        conv_i = 0
        for layer in vgg.children():
            if   isinstance(layer, nn.Conv2d):    conv_i += 1; name = f"conv_{conv_i}"
            elif isinstance(layer, nn.ReLU):      name = f"relu_{conv_i}"; layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d): name = f"pool_{conv_i}"
            elif isinstance(layer, nn.BatchNorm2d): name = f"bn_{conv_i}"
            else: continue
            self.layers.append(layer)
            self.names.append(name)

    def forward(self, x):
        style_feats, content_feats = {}, {}
        for layer, name in zip(self.layers, self.names):
            x = layer(x)
            if name in self.style_layers:   style_feats[name]   = x
            if name in self.content_layers: content_feats[name] = x
        return style_feats, content_feats

# -------------------------
# 3. Transformer network
# -------------------------
class ConvINReLU(nn.Module):
    def __init__(self, in_c, out_c, k, stride):
        super().__init__()
        pad = k // 2
        self.seq = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, stride, pad),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.seq(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvINReLU(channels, channels, 3, 1),
            ConvINReLU(channels, channels, 3, 1)
        )
    def forward(self, x): return x + self.block(x)

class UpsampleConvINReLU(nn.Module):
    def __init__(self, in_c, out_c, k, scale):
        super().__init__()
        pad = k // 2
        self.seq = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='nearest'),
            nn.Conv2d(in_c, out_c, k, 1, pad),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.seq(x)

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # downsampling
        self.conv1 = ConvINReLU(3,   32, 9, 1)
        self.conv2 = ConvINReLU(32,  64, 3, 2)
        self.conv3 = ConvINReLU(64, 128, 3, 2)
        # residuals
        self.res  = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        # upsampling
        self.deconv1 = UpsampleConvINReLU(128, 64, 3, 2)
        self.deconv2 = UpsampleConvINReLU(64,  32, 3, 2)
        self.conv_out = nn.Sequential(
            nn.Conv2d(32, 3, 9, 1, 4),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.res(y)
        y = self.deconv1(y)
        y = self.deconv2(y)
        return self.conv_out(y)

# -------------------------
# 4. Loss modules
# -------------------------
class TVLoss(nn.Module):
    def forward(self, x):
        b,c,h,w = x.size()
        tv_h = torch.pow(x[:,:,1:,:] - x[:,:,:h-1,:], 2).sum()
        tv_w = torch.pow(x[:,:,:,1:] - x[:,:,:,:w-1], 2).sum()
        return tv_h + tv_w

"./data/"
# -------------------------
# 5. Average style grams
# -------------------------
def compute_average_grams(style_dir, loader, vgg, device, style_layers):
    pattern = os.path.join(style_dir, "**", "*.[jp][pn]g")
    paths = glob(pattern, recursive=True)
    if not paths:
        raise RuntimeError("No style images found.")
    # init accumulator
    feats, _ = vgg(loader(Image.open(paths[0]).convert("RGB")).unsqueeze(0).to(device))
    avg = {L: torch.zeros_like(gram_matrix(feats[L])) for L in feats}
    with torch.no_grad():
        for p in paths:
            img = loader(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            s_feats, _ = vgg(img)
            for L in s_feats:
                avg[L] += gram_matrix(s_feats[L])
    n = len(paths)
    return {L: avg[L] / n for L in avg}


# -------------------------
# 6. Training loop
# -------------------------
def train(args):
    # Device & logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    # Transforms & loader
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
    loader = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize
    ])
    dataset = datasets.ImageFolder(args.content_dir, transform=loader)
    loader_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Log dataset/iteration info
    num_batches = len(loader_dl)
    total_iters = num_batches * args.epochs
    logger.info(
        f"Content: {len(dataset)} images, batch={args.batch_size} → "
        f"{num_batches} batches/epoch, epochs={args.epochs} → "
        f"{total_iters} total updates"
    )

    # VGG & style grams
    style_layers   = [f"conv_{i}" for i in range(1, 6)]
    content_layers = ["conv_4"]
    vgg = VGGFeatures(style_layers, content_layers).to(device)
    avg_grams = compute_average_grams(
        args.style_dir, loader, vgg, device, style_layers
    )

    # Transformer & optimizer
    transformer = TransformerNet().to(device)
    optimizer   = optim.Adam(transformer.parameters(), lr=args.lr)

    # Resume from checkpoint
    if os.path.isfile(args.model_out):
        logger.info(f"Loading checkpoint '{args.model_out}'")
        chk = torch.load(args.model_out, map_location=device)
        transformer.load_state_dict(chk)
        logger.info("Checkpoint loaded; continuing training")

    # Prepare denorm + PIL converter for snapshots
    inv_mean = [-m/s for m, s in zip([0.485,0.456,0.406], [0.229,0.224,0.225])]
    inv_std  = [1.0/s for s in [0.229,0.224,0.225]]
    denorm   = transforms.Normalize(mean=inv_mean, std=inv_std)
    to_pil   = transforms.ToPILImage()

    # Loss weights
    style_weight = 500.0
    style_w = {L: style_weight/len(style_layers) for L in style_layers}
    tv_weight = 1e-6
    tv_loss_fn = TVLoss()

    iteration = 0
    for epoch in range(1, args.epochs + 1):
        transformer.train()
        total_loss = 0.0

        loop = tqdm(loader_dl, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch, _ in loop:
            iteration += 1
            batch = batch.to(device)
            optimizer.zero_grad()

            out = transformer(batch)
            s_out, c_out = vgg(out)
            _, c_orig   = vgg(batch)

            # Style loss
            style_loss = 0.0
            for L in style_layers:
                G_out = gram_matrix(s_out[L])
                style_loss += style_w[L] * nn.functional.mse_loss(G_out, avg_grams[L])

            # Content & TV loss
            content_loss = nn.functional.mse_loss(c_out["conv_4"], c_orig["conv_4"])
            tv_loss      = tv_weight * tv_loss_fn(out)

            loss = style_loss + content_loss + tv_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(
                style=f"{style_loss.item():.4f}",
                content=f"{content_loss.item():.4f}",
                tv=f"{tv_loss.item():.4f}"
            )

            # Save a stylized snapshot every 500 updates
            if iteration % 500 == 0:
                img_t = out[0].detach().cpu()           # pick first image of batch
                img_t = (img_t + 1.0) * 0.5             # from [-1,1] → [0,1]
                img_t = denorm(img_t).clamp(0, 1)       # undo ImageNet norm
                pil = to_pil(img_t)
                os.makedirs("checkpoints", exist_ok=True)
                pil.save(f"checkpoints/epoch{epoch}_iter{iteration}.png")

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch}/{args.epochs} — avg loss: {avg_loss:.4f}")

    # Save final model
    torch.save(transformer.state_dict(), args.model_out)
    logger.info(f"Model saved to {args.model_out}")
    
# -------------------------
# 7. Inference helper
# -------------------------
def stylize_image(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(args.model_out, map_location=device))
    transformer.eval()

    loader   = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    unloader = transforms.ToPILImage()

    # build a list of image paths
    if os.path.isdir(args.content_dir):
        # all files in the folder
        paths = [
            os.path.join(args.content_dir, f)
            for f in os.listdir(args.content_dir)
            if f.lower().endswith((".jpg","jpeg","png"))
        ]
    elif os.path.isfile(args.content_dir):
        # single image
        paths = [args.content_dir]
    else:
        raise RuntimeError(f"No such file or directory: {args.content_dir}")

    for path in paths:
        img = Image.open(path).convert("RGB")
        input_t = loader(img).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = transformer(input_t)          # in [-1, +1]
            out_t = (raw + 1.) * 0.5            # now in [0,1]
            out_t = out_t.clamp(0,1)            # just guard against tiny overshoot
        out_img = unloader(out_t.cpu().squeeze(0))

        # name the output by original filename
        base = os.path.basename(path)
        out_img.save(f"./data/output4/stylized_full{base}")
        print(f"Saved stylized_{base}")
# -------------------------
# 8. Argument parsing
# -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--content_dir", type=str, required=True,
        help="Directory (or single image) for content / inference."
    )
    p.add_argument(
        "--style_dir", type=str, default=None,
        help="Directory of style images; only required for training."
    )
    p.add_argument(
        "--model_out", type=str, default="transformer.pth",
        help="Where to save (or load) the transformer model."
    )
    p.add_argument(
        "--epochs", type=int, default=2,
        help="Number of training epochs (training only)."
    )
    p.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for training."
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for training."
    )
    p.add_argument(
        "--image_size", type=int, default=256,
        help="Resize / crop size for both train & stylize."
    )
    p.add_argument(
        "--stylize", action="store_true",
        help="Run inference only (no training)."
    )
    args = p.parse_args()

    if args.stylize:
        # Inference mode: style_dir and training params are ignored
        stylize_image(args)
    else:
        # Training mode: style_dir *must* be provided
        if args.style_dir is None:
            p.error("--style_dir is required unless --stylize is set")
        train(args)
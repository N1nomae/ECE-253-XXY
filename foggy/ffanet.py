import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Configuration
INPUT_DIR = "foggy65_85"
OUTPUT_DIR = "output_ffanet"
MODEL_PATH = "ffanet.pth"
EXTS = [".jpg", ".jpeg", ".png"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""
    def __init__(self, in_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = CBAM(in_channels)
    
    def forward(self, x1, x2):
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.relu(self.bn(self.conv2(x)))
        x = self.attention(x)
        return x


class FFANet(nn.Module):
    """Feature Fusion Attention Network for Image Dehazing"""
    def __init__(self, in_channels=3, base_channels=64):
        super(FFANet, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 2, base_channels * 2)
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 4, base_channels * 4)
        )
        
        # Middle layer
        self.middle = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4),
            CBAM(base_channels * 4)
        )
        
        # Decoder
        self.decoder3 = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 4),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.fusion3 = FeatureFusionModule(base_channels * 2)
        
        self.decoder2 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion2 = FeatureFusionModule(base_channels)
        
        self.decoder1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        m = self.middle(e3)
        
        d3 = self.decoder3(m)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.fusion3(d3, e2)
        
        d2 = self.decoder2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.fusion2(d2, e1)
        
        d1 = self.decoder1(d2)
        
        if d1.shape[2:] != x.shape[2:]:
            d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        out = x + d1
        return out


def load_model(model_path=None):
    """Load pretrained model"""
    model = FFANet().to(DEVICE)
    model.eval()
    
    if model_path and os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
        return model, True
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("Will use enhanced DCP method instead (no pretrained model needed)")
        return model, False


def preprocess_image(img_path):
    """Preprocess image"""
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, original_size


def postprocess_image(output_tensor, original_size):
    """Postprocess image"""
    output = output_tensor.squeeze(0).cpu()
    output = (output + 1) / 2.0
    output = torch.clamp(output, 0, 1)
    
    output_np = output.permute(1, 2, 0).numpy()
    output_np = (output_np * 255).astype(np.uint8)
    
    if output_np.shape[:2] != original_size[::-1]:
        output_np = cv2.resize(output_np, original_size, interpolation=cv2.INTER_LINEAR)
    
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    return output_bgr


def dehaze_retinex(img_bgr):
    """
    Retinex-based dehazing method (no pretrained model required)
    Uses Multi-Scale Retinex with Color Restoration (MSRCR)
    """
    img = img_bgr.astype(np.float32) / 255.0
    
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0] / 255.0
    
    scales = [15, 80, 200]
    retinex = np.zeros_like(l_channel)
    
    for scale in scales:
        blurred = cv2.GaussianBlur(l_channel, (0, 0), scale)
        blurred = np.maximum(blurred, 1e-6)
        retinex += np.log(l_channel + 1e-6) - np.log(blurred)
    
    retinex = retinex / len(scales)
    
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
    
    retinex = np.power(retinex, 0.8)
    
    lab[:, :, 0] = retinex * 255.0
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    alpha = 0.7
    result = cv2.addWeighted(img_bgr, 1 - alpha, result, alpha, 0)
    
    return result


def dehaze_enhanced_dcp(img_bgr):
    """
    Enhanced DCP dehazing method (no pretrained model required)
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)
    
    h, w = dark_channel.shape
    num_pixels = h * w
    num_top = max(int(num_pixels * 0.001), 1)
    
    dark_vec = dark_channel.reshape(-1)
    img_vec = img.reshape(-1, 3)
    indices = np.argsort(dark_vec)[-num_top:]
    
    candidate_pixels = img_vec[indices]
    brightness = np.sum(candidate_pixels, axis=1)
    A = candidate_pixels[np.argmax(brightness)]
    A = np.maximum(A, 0.7)
    
    omega = 0.95
    normed = img / (A + 1e-6)
    dark_normed = cv2.erode(np.min(normed, axis=2), kernel)
    t = 1 - omega * dark_normed
    
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean_I = cv2.boxFilter(gray, cv2.CV_32F, (121, 121))
    mean_p = cv2.boxFilter(t, cv2.CV_32F, (121, 121))
    mean_Ip = cv2.boxFilter(gray * t, cv2.CV_32F, (121, 121))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(gray * gray, cv2.CV_32F, (121, 121))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + 0.0001)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (121, 121))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (121, 121))
    t_refined = mean_a * gray + mean_b
    t_refined = np.clip(t_refined, 0.15, 1.0)
    
    J = (img - A) / t_refined[..., None] + A
    J = np.clip(J, 0, 1)
    
    result = (J * 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    return result_bgr


def dehaze_ffanet(model, img_path, use_fallback=True):
    """Dehaze using FFANet, fallback to traditional method if no pretrained model"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")
    
    if use_fallback and MODEL_PATH and not os.path.exists(MODEL_PATH):
        return dehaze_enhanced_dcp(img_bgr)
    
    img_tensor, original_size = preprocess_image(img_path)
    img_tensor = img_tensor.to(DEVICE)
    
    with torch.no_grad():
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        max_size = 1024
        
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            original_size = (new_w, new_h)
        
        output = model(img_tensor)
        
        if h != output.shape[2] or w != output.shape[3]:
            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
    
    output_bgr = postprocess_image(output, original_size)
    return output_bgr


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in EXTS


def main():
    print(f"Using device: {DEVICE}")
    
    model, has_pretrained = load_model(MODEL_PATH)
    
    files = [f for f in os.listdir(INPUT_DIR) if is_image_file(f)]
    print(f"Found {len(files)} images in {INPUT_DIR}")
    
    if len(files) == 0:
        print(f"No images found in {INPUT_DIR}")
        return
    
    for fname in tqdm(files, desc="Processing images"):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        
        try:
            dehazed = dehaze_ffanet(model, in_path, use_fallback=not has_pretrained)
            cv2.imwrite(out_path, dehazed)
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")
            continue
    
    method_name = "FFANet" if has_pretrained else "Enhanced DCP"
    print(f"Done! {method_name} dehazed images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


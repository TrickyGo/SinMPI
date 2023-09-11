from PIL import Image, ImageOps
import cv2
import torch
import torchvision
from torchvision import transforms

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        self.mean.requires_grad = False
        self.std.requires_grad = False
        self.resize = resize

    def forward(self, syn_imgs, gt_imgs):
        syn_imgs = (syn_imgs - self.mean) / self.std
        gt_imgs = (gt_imgs - self.mean) / self.std
        if self.resize:
            syn_imgs = self.transform(syn_imgs, mode="bilinear", size=(224, 224),
                                      align_corners=False)
            gt_imgs = self.transform(gt_imgs, mode="bilinear", size=(224, 224),
                                     align_corners=False)

        loss = 0.0
        x = syn_imgs
        y = gt_imgs
        for block in self.blocks:
            with torch.no_grad():
                x = block(x)
                y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def image_to_tensor(img_path, unsqueeze=True):
    im = Image.open(img_path).convert('RGB')
    if img_path[-3:] == 'jpg':
        im = ImageOps.exif_transpose(im)
    rgb = transforms.ToTensor()(im)
    if unsqueeze:
        rgb = rgb.unsqueeze(0)
    return rgb

def disparity_to_tensor(disp_path, unsqueeze=True):
    disp = cv2.imread(disp_path, -1) / (2 ** 16 - 1)
    disp = torch.from_numpy(disp)[None, ...]
    if unsqueeze:
        disp = disp.unsqueeze(0)
    return disp.float()

def tensor_to_depth(tensor): # BCHW
    model_type = "DPT_Large" 
    midas_model = torch.hub.load("/home/pug/.cache/torch/hub/MiDaS", model_type, source='local').cuda()

    midas_transforms = torch.hub.load("/home/pug/.cache/torch/hub/MiDaS", "transforms", source='local')
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    input_batch = tensor
    with torch.no_grad():
        prediction = midas_model(input_batch)
    output = prediction
    output = (output - output.min()) / (output.max() - output.min())
    return output.unsqueeze(0)

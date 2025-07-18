"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import time
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision
import imgaug.augmenters as iaa
import torchvision.transforms as transforms
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = ((x1 + x2) / 2)
            boxes[box_idx, 2] = ((y1 + y2) / 2)
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return img, boxes
    
class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes


class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets
class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes

DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes
    
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def to_cpu(tensor):
    return tensor.detach().cpu()

def box_iou(box1, box2, eps=1e-7):
    # box1: [N, 4], box2: [M, 4] => return: [N, M]
    inter = (
        torch.min(box1[:, None, 2:], box2[:, 2:]) -
        torch.max(box1[:, None, :2], box2[:, :2])
    ).clamp(0).prod(2)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter + eps

    return inter / union

def soft_nms(prediction, conf_thres=0.25, iou_thres=0.45, sigma=0.5, classes=None):
    nc = prediction.shape[2] - 5
    device = prediction.device
    output = [torch.zeros((0,6), device=device) for _ in range(prediction.shape[0])]

    for img_i, preds in enumerate(prediction):
        preds = preds[preds[:, 4] > conf_thres]
        if preds.shape[0] == 0:
            continue

        preds = preds.clone()  # 避免原地修改
        scores = preds[:, 5:] * preds[:, 4:5]  # 这里新建scores，不修改preds

        boxes = xywh2xyxy(preds[:, :4])

        if nc > 1:
            i, j = (scores > conf_thres).nonzero(as_tuple=False).T
            scores = scores[i, j]
            boxes = boxes[i]
            classes_ = j.float()
        else:
            scores, j = scores.max(1)
            keep = scores > conf_thres
            boxes = boxes[keep]
            scores = scores[keep]
            classes_ = j[keep].float()

        if classes is not None:
            cls_mask = torch.isin(classes_, torch.tensor(classes, device=device))
            boxes = boxes[cls_mask]
            scores = scores[cls_mask]
            classes_ = classes_[cls_mask]

        if boxes.shape[0] == 0:
            continue

        keep_boxes = []
        scores_ = scores.clone()
        keep_mask = torch.ones(boxes.shape[0], dtype=torch.bool, device=device)

        while keep_mask.any():
            masked_scores = scores_.clone()
            masked_scores = masked_scores.masked_fill(~keep_mask, -1)
            max_idx = torch.argmax(masked_scores)
            max_box = boxes[max_idx].unsqueeze(0)
            max_score = scores_[max_idx]
            max_cls = classes_[max_idx]

            keep_boxes.append(torch.cat([max_box.squeeze(), max_score.unsqueeze(0), max_cls.unsqueeze(0)]))

            old_keep_mask = keep_mask.clone()
            old_keep_mask[max_idx] = False

            if old_keep_mask.sum() == 0:
                break

            rest_boxes = boxes[old_keep_mask]
            ious = box_iou(max_box, rest_boxes).squeeze(0)
            new_scores = scores_[old_keep_mask] * torch.exp(-(ious ** 2) / sigma)

            # 使用 torch.where 来避免原地赋值
            new_keep_mask = torch.where(new_scores > conf_thres, torch.tensor(True, device=device), torch.tensor(False, device=device))

            # 更新 keep_mask，构造新的 mask
            temp_mask = old_keep_mask.clone()
            idxs = torch.nonzero(old_keep_mask, as_tuple=False).squeeze(1)
            temp_mask[idxs] = new_keep_mask
            keep_mask = temp_mask

            # 更新 scores_, 也用 torch.where 创建新 tensor
            updated_scores = scores_.clone()
            final_idx = idxs[new_keep_mask]
            for idx, new_score in zip(final_idx, new_scores[new_keep_mask]):
                updated_scores[idx] = new_score
            scores_ = updated_scores

        if keep_boxes:
            output[img_i] = torch.stack(keep_boxes)

    return output




# def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
#     """Performs Non-Maximum Suppression (NMS) on inference results
#     Returns:
#         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
#     """
#     nc = prediction.shape[2] - 5  # number of classes

#     # Settings
#     # (pixels) minimum and maximum box width and height
#     max_wh = 4096
#     max_det = 300  # maximum number of detections per image
#     max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
#     time_limit = 1.0  # seconds to quit after
#     multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

#     t = time.time()
#     output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         x = x[x[..., 4] > conf_thres]  # confidence

#         # If none remain process next image
#         if not x.shape[0]:
#             continue

#         # Compute conf
#         x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

#         # Box (center x, center y, width, height) to (x1, y1, x2, y2)
#         box = xywh2xyxy(x[:, :4])

#         # Detections matrix nx6 (xyxy, conf, cls)
#         if multi_label:
#             i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
#             x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
#         else:  # best class only
#             conf, j = x[:, 5:].max(1, keepdim=True)
#             x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

#         # Filter by class
#         if classes is not None:
#             x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

#         # Check shape
#         n = x.shape[0]  # number of boxes
#         if not n:  # no boxes
#             continue
#         elif n > max_nms:  # excess boxes
#             # sort by confidence
#             x = x[x[:, 4].argsort(descending=True)[:max_nms]]

#         # Batched NMS
#         c = x[:, 5:6] * max_wh  # classes
#         # boxes (offset by class), scores
#         boxes, scores = x[:, :4] + c, x[:, 4]
#         i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
#         if i.shape[0] > max_det:  # limit detections
#             i = i[:max_det]

#         output[xi] = to_cpu(x[i])

#         if (time.time() - t) > time_limit:
#             print(f'WARNING: NMS time limit {time_limit}s exceeded')
#             break  # time limit exceeded

#     return output

# def rescale_boxes(boxes, current_dim, original_shape):
#     """
#     Rescales bounding boxes to the original shape
#     """
#     orig_h, orig_w = original_shape

#     # The amount of padding that was added
#     pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
#     pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

#     # Image height and width after padding is removed
#     unpad_h = current_dim - pad_y
#     unpad_w = current_dim - pad_x

#     # Rescale bounding boxes to dimension of original image
#     boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
#     boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
#     boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
#     boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
#     return boxes

def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original image shape using fully differentiable PyTorch operations.

    Args:
        boxes (Tensor): shape (N, 4) in format (x1, y1, x2, y2) normalized to current_dim.
        current_dim (int or float): the dimension of the input image to the model, e.g., 416.
        original_shape (tuple): original (height, width) of the image.

    Returns:
        Tensor: rescaled boxes of shape (N, 4) mapped to original image size.
    """
    # Ensure current_dim is tensor
    if not isinstance(current_dim, torch.Tensor):
        current_dim = torch.tensor(current_dim, dtype=boxes.dtype, device=boxes.device)

    # Convert original H, W to tensor
    orig_h = torch.tensor(original_shape[0], dtype=boxes.dtype, device=boxes.device)
    orig_w = torch.tensor(original_shape[1], dtype=boxes.dtype, device=boxes.device)

    max_orig = torch.max(orig_h, orig_w)
    scale = current_dim / max_orig

    pad_x = (orig_h - orig_w).clamp(min=0) * scale
    pad_y = (orig_w - orig_h).clamp(min=0) * scale

    unpad_w = current_dim - pad_x
    unpad_h = current_dim - pad_y

    # Fully differentiable padding removal and rescaling
    boxes_x1 = ((boxes[:, 0] - pad_x / 2) / unpad_w) * orig_w
    boxes_y1 = ((boxes[:, 1] - pad_y / 2) / unpad_h) * orig_h
    boxes_x2 = ((boxes[:, 2] - pad_x / 2) / unpad_w) * orig_w
    boxes_y2 = ((boxes[:, 3] - pad_y / 2) / unpad_h) * orig_h

    return torch.stack([boxes_x1, boxes_y1, boxes_x2, boxes_y2], dim=1)
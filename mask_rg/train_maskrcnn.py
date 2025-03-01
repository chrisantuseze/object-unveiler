import torch
import torchvision
from mask_rg.segmentation_dataset import SegmentationDataset
import utils.log_utils as log_utils
import utils.torch_utils as torch_utils
import datetime
import argparse
import time
import os
import numpy as np

from mask_rg.coco_utils import get_coco_api_from_dataset
from mask_rg.coco_eval import CocoEvaluator
import mask_rg.transforms as T
import math
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import sys
from PIL import Image


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True, trainable_backbone_layers=5
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_data_loader(dataset_root, batch_size, is_real):
    # use our dataset and defined transformations
    dataset = SegmentationDataset(dataset_root, get_transform(train=True), is_real)
    dataset_test = SegmentationDataset(dataset_root, get_transform(train=False), is_real)

    # split the dataset in train and test set
    indices = torch.arange(0, len(dataset)).tolist()
    test_point = -100
    dataset_sub_train = torch.utils.data.Subset(dataset, indices[:test_point])
    dataset_sub_test = torch.utils.data.Subset(dataset_test, indices[test_point:])

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_sub_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=torch_utils.collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_sub_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=torch_utils.collate_fn,
    )

    return data_loader_train, data_loader_test


def train_one_epoch(model, optimizer, data_loader, device, epoch, logger, print_freq, resume=False):
    """
    https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    """
    model.train()
    metric_logger = log_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", log_utils.SmoothedValue(window_size=1, fmt="{value:.8f}"))
    header = "Epoch: [{}]".format(epoch)

    lr_scheduler = None
    if epoch == 0 and not resume:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, logger, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = torch_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, logger, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = log_utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, logger, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


@torch.no_grad()
def test(model, data_loader, device):
    model.eval()

    print(len(data_loader.dataset))
    count = 0
    for images, targets in data_loader:
        for i in range(len(images)):
            print(targets[i]["image_id"])
            image = images[i]

            # target = targets[i]
            # image = image.permute(1, 2, 0).numpy()
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # image *= 255
            # image = image.astype(np.uint8)
            # cv2.imwrite(str(i) + 'color.png', image)
            # masks = target['masks']
            # for mi, m in enumerate(masks):
            #     img = m.numpy()
            #     img *= 255
            #     print(np.max(img), np.min(img))
            #     img = img.astype(np.uint8)
            #     cv2.imwrite(str(i+mi) + 'mask.png', img)
            # exit(5)

            prediction = model([image.to(device)])
            print(prediction[0]["scores"])
            pred_mask = np.zeros((224, 224), dtype=np.uint8)
            if len(targets[i]["masks"]) != np.sum(prediction[0]["scores"].cpu().numpy() > 0.9):
                for idx, mask in enumerate(prediction[0]["masks"]):
                    if prediction[0]["scores"][idx] > 0.9:
                        # if prediction[0]['scores'][idx] > 0.75:
                        img1 = mask[0].mul(255).byte().cpu().numpy()
                        img1[img1 > 80] = 255
                        img1[img1 <= 80] = 0
                        pred_mask[img1 > 80] = 255 - idx * 5

                        img1 = Image.fromarray(img1)
                        img1.save(str(count) + "-" + str(idx) + "mask.png")
                for idx, mask in enumerate(targets[i]["masks"]):
                    img2 = Image.fromarray(mask.mul(255).byte().cpu().numpy())
                    img2.save(str(count) + "-" + str(idx) + "target.png")
                print(len(targets[i]["masks"]), len(prediction[0]["masks"] > 0.9))
                img0 = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
                img0.save(str(count) + "-" + str(idx) + "ori.png")
                img0 = Image.fromarray(pred_mask)
                img0.save(str(count) + "-" + str(idx) + "pred.png")
                count += 1
                exit()


def main(args):
    data_loader, data_loader_test = get_data_loader(
        args.dataset_root, args.batch_size, args.is_real
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model_instance_segmentation(2)
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.dataset_root, "maskrcnn.pth"), map_location=device))
    model = model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by 10x every 1 epochs
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3], gamma=0.1)

    log_dir = os.path.join(args.dataset_root, "runs")
    timestamp_value = datetime.datetime.fromtimestamp(time.time())
    time_name = timestamp_value.strftime("%Y-%m-%d-%H-%M")
    log_dir = os.path.join(log_dir, time_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = log_utils.setup_logger(log_dir, "Mask R-CNN")

    if args.test:
        test(model, data_loader_test, device=device)
    else:
        for epoch in range(args.epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(
                model,
                optimizer,
                data_loader,
                device,
                epoch,
                logger,
                print_freq=50,
                resume=args.resume,
            )
            torch.save(model.state_dict(), os.path.join(args.dataset_root, "maskrcnn.pth"))
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, logger, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train foreground")

    parser.add_argument(
        "--dataset_root", dest="dataset_root", action="store", help="Enter the path to the dataset"
    )
    parser.add_argument("--is_real", dest="is_real", action="store_true", default=False, help="")
    parser.add_argument(
        "--epochs",
        dest="epochs",
        action="store",
        type=int,
        default=5,
        help="Enter the epoch for training",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        action="store",
        type=int,
        default=6,
        help="Enter the batchsize for training and testing",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", default=False, help="Testing and visualizing"
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=False,
        help="Enter the path to the dataset",
    )

    args = parser.parse_args()
    if args.test:
        args.resume = True

    main(args)
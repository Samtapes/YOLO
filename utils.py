import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4] # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)



def non_max_suppression(
    bboxes,
    iou_treshold,
    threshold,
    box_format="corners"
):
    """
    Remove extra bboxes from predictions.

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2] with shape: [S * S, 6]
        iou_threshold (float): threshold where predict bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or corners used to specify bboxes

    Returns:
        list: bboxes after NMS given a specified IoU threshold 
        with shape: [S * S, class_num, prob, x, y, w, h]
    """

    # pred.shape: [S * S, class_num, prob, x, y, w, h]
    # predictions = [[1, 0.9, x1, y1, x2, y2]]
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            )
            < iou_treshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms



def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20
):
    # pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes): # Repeating for all classes
        detections = []
        ground_truths = []

        for detection in pred_boxes: # Selecting all boxes from the same class
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        # Counting how many target boxes each image have

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # amount_bboxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}        
        for key, val in amount_bboxes.items(): # Looping all images
            amount_bboxes[key] = torch.zeros(val)


        # Sorting all predicted boxes from probability
        detections.sort(key=lambda x: x[2], reverse=True) 
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections): # Looping all predicted bounding boxes
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ] # Selecting all ground truths from the same image of that predicted bounding box

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img): # Looping all target bounding box in the image same image
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                ) # Calculating IOU

                if iou > best_iou: # Saving IOU and best predicted bounding box idx
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # Only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0: # If you haven't found this bounding box
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # If IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor[1]), precisions)
        recalls = torch.cat((torch.tensor[0]), recalls)
        average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """
    Plots predicted bounding boxes on the image.

    Receives image and bboxes.
    
    bboxes (list): bboxes with non max suppression applied of a single image, with shape:
    [selected_boxes, 6]
    """

    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes: # Percorrendo todas as células
        box = box[2:] # Retirando a classe e a probabilidade da caixa
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show() 


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold, # Prop threshold
    pred_format="cells",
    box_format="midpoint",
    device="cuda"
):
    """
    Go to all examples and get pred boxes.
    Convert all bboxes relative to the cell to be relative to the image.
    Apply non max suppression to predicted bounding boxes

    Returns: all_pred_boxes, all_true_boxes
    shape: (train_idx, class_pred, prob_score, x1, y1, x2, y2)
    
    """

    all_pred_boxes = []
    all_trues_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)
            # Pred shape: [N, 1470] -> [N, S, S, 30]

        # Converting bboxes relative to cell to image
        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions) # Returns the bbox with the best probability
        # list of shape: [N, S * S, class_pred, prob_score, x1, y1, w, h]

        # Looping all individual images
        for idx in range(batch_size):

            # Applying nms to the predictions
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_treshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            ) # nms_boxes.shape: [selected_boxes, 6]
        
            #if batch_idx == 0 and idx == 0:
            #   plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #   print(nms_boxes)

            # Adding idx to pred bboxes
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            
            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_trues_boxes.append([train_idx] + box)
        
            train_idx += 1
    
    model.train()
    return all_pred_boxes, all_trues_boxes


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.
    """

    predictions = predictions.to("cpu")
    # shape: [N, 1470] -> [N, S, S, 30]
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, 30)

    # Getting both bboxes
    bboxes1 = predictions[..., 21:25] # shape: [N, S, S, 4]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    # predictions[..., 20].shape = [N, S, S, 1]
    # scores shape = [2, N, S, S, 1]
    best_box = scores.argmax(0).unsqueeze(-1)
    # shape: [N, S, S, 1], and the last dim is a num between 0 and 1.
    # 0, the best prob is fot the 1st box, and 1, best prob for the 2nd box
    best_boxes = bboxes1 * (1 - best_box) + best_boxes * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    # shape: [N, S, S, 1]
    # Converting bboxes
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1) # shape: [N, S, S, 4]
    predicted_class = predictions[..., :20].argmax(-1).usqueeze(-1) # Getting the class number. shape [N, S, S, 1]
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    ) # shape: [N, S, S, 6] -> [N, S, S, class_num, prob, x, y, w, h]

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    """
    Converts tensor to python list and changes
    matrix shape.

    Returns list of shape: [N, S * S, 6]
    """

    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    # Original shape were [N, S, S, 6] and now is [N, S * S, 6]
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_boxes = []

    # Looping all the "images"
    for ex_idx in range(out.shape[0]):
        bboxes = []

        # Looping every cell
        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]]) # x.shape: [6]
        # bboxes.shape: [S * S, 6]

        all_boxes.append(bboxes)

    # all_boxes.shape: [N, S * S, 6]
    return all_boxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
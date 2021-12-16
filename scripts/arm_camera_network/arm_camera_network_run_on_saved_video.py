import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import tqdm
import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import os
from sensor_msgs.msg import Image, RegionOfInterest
from odhe_ros.msg import Result
from cv_bridge import CvBridge
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# Extract video properties
video = cv2.VideoCapture('/home/labuser/situation9.avi')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

br = CvBridge()

# Initialize video writer
video_writer = cv2.VideoWriter('out_situation9.mp4', fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

register_coco_instances("train_set", {}, "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/train/annotations.json", "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/train")
train_metadata = MetadataCatalog.get("train_set")

# Initialize predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_set")
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.MAX_ITER = 1000 # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # 5 classes (Plate, Carrot, Celery, Pretzel, Gripper)

# Temporary Solution. If I train again I think I can use the dynamically set path again
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/output/model_final.pth")
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("test_set")
predictor = DefaultPredictor(cfg)

class_names = ['plate', 'carrot', 'celery', 'pretzel', 'gripper']

# Colors = [blue, green, red]
color_plate = [0, 255, 0]       # green
color_carrot = [255, 200, 0]    # blue
color_celery = [0, 0, 255]      # red
color_pretzel = [0, 220, 255]   # yellow
color_gripper = [204, 0, 150]   # purple
colors = list([color_plate, color_carrot, color_celery, color_pretzel, color_gripper])

alpha = .4

# Initialize visualizer
v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

def getResult(predictions, classes):

    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        #print(type(masks))
    else:
        return

    result_msg = Result()
    # result_msg.header = self._header
    result_msg.class_ids = predictions.pred_classes if predictions.has("pred_classes") else None
    result_msg.class_names = np.array(classes)[result_msg.class_ids.numpy()]
    result_msg.scores = predictions.scores if predictions.has("scores") else None

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        mask = np.zeros(masks[i].shape, dtype="uint8")
        mask[masks[i, :, :]]=255
        mask = br.cv2_to_imgmsg(mask)
        result_msg.masks.append(mask)

        box = RegionOfInterest()
        box.x_offset = np.uint32(x1)
        box.y_offset = np.uint32(y1)
        box.height = np.uint32(y2 - y1)
        box.width = np.uint32(x2 - x1)
        result_msg.boxes.append(box)

    return result_msg

def runOnVideo(video, maxFrames):
    """ Runs the predictor on every frame in the video (unless maxFrames is given),
    and returns the frame with the predictions drawn.
    """

    readFrames = 0
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        # Get prediction results for this frame
        outputs = predictor(frame)
        predictions = outputs["instances"].to("cpu")

        result = getResult(predictions, class_names)

        # Visualize using custom cv2 code
        if result is not None:
            result_cls = result.class_names
            result_clsId = result.class_ids
            result_scores = result.scores
            result_masks = result.masks

            # Create copies of the original image
            im = frame.copy()
            output = frame.copy()

            # Initialize lists
            masks = []
            masks_indices = []
            for i in range(len(result_clsId)):
                # Obtain current object mask as a numpy array (black and white mask of single object)
                current_mask = br.imgmsg_to_cv2(result_masks[i])

                # Find current mask indices
                mask_indices = np.where(current_mask==255)

                # Add to mask indices list
                if len(masks_indices) > len(result_clsId):
                    masks_indices = []
                else:
                    masks_indices.append(mask_indices)

                # Add to mask list
                if len(masks) > len(result_clsId):
                    masks = []
                else:
                    masks.append(current_mask)

            if len(masks) > 0:
                # Create composite mask
                composite_mask = sum(masks)

                # Clip composite mask between 0 and 255   
                composite_mask = composite_mask.clip(0, 255)

            # # Apply mask to image
            # masked_img = cv2.bitwise_and(im, im, mask=current_mask)

            # Find indices of object in mask
            # composite_mask_indices = np.where(composite_mask==255)

            for i in range(len(result_clsId)):
                # Select correct object color
                color = colors[result_clsId[i]]

                # Change the color of the current mask object
                im[masks_indices[i][0], masks_indices[i][1], :] = color

            # Apply alpha scaling to image to adjust opacity
            cv2.addWeighted(im, alpha, output, 1 - alpha, 0, output)

            for i in range(len(result_clsId)):
                # Draw Bounding boxes
                start_point = (result.boxes[i].x_offset, result.boxes[i].y_offset)
                end_point = (result.boxes[i].x_offset + result.boxes[i].width, result.boxes[i].y_offset + result.boxes[i].height)
                start_point2 = (result.boxes[i].x_offset + 2, result.boxes[i].y_offset + 2)
                end_point2 = (result.boxes[i].x_offset + result.boxes[i].width - 2, result.boxes[i].y_offset + 12)
                color = colors[result_clsId[i]]
                box_thickness =  1

                name = result_cls[i]
                score = result_scores[i]
                conf = round(score.item() * 100, 1)
                string = str(name) + ":" + str(conf) + "%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (result.boxes[i].x_offset + 2, result.boxes[i].y_offset + 10)
                fontScale = .3
                text_thickness = 1
                output = cv2.rectangle(output, start_point, end_point, color, box_thickness)
                output = cv2.rectangle(output, start_point2, end_point2, color, -1)     # Text box
                output = cv2.putText(output, string, org, font, fontScale, [0, 0, 0], text_thickness, cv2.LINE_AA, False)

            readFrames += 1
            if readFrames > maxFrames:
                break

        yield output

        

# Create a cut-off for debugging
num_frames = 600

# Enumerate the frames of the video
for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):

    # Write test image
    # cv2.imwrite('out_situation3.png', visualization)

    # Write to video file
    video_writer.write(visualization)

# Release resources
video.release()
video_writer.release()
cv2.destroyAllWindows()
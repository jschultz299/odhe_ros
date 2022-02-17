import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2, random
import time
import pyrealsense2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

import rospy
import sys
from cv_bridge import CvBridge
from odhe_ros.msg import Result
from gpd_ros.msg import CloudSamples, CloudSources
from sensor_msgs.msg import Image, RegionOfInterest, PointCloud2, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Int64

class maskRCNN(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pc_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camInfo_callback)

        # Publishers
        self.pub = rospy.Publisher('arm_camera_objects', Image, queue_size=10)
        self.result_pub = rospy.Publisher('arm_camera_results', Result, queue_size=10)
        self.pc_pub = rospy.Publisher('cloud_samples', CloudSamples, queue_size=10)

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def pc_callback(self, msg):
        self.pointCloud = msg

    def depth_callback(self, msg):
        self.depth_array = self.convert_depth_image(msg)
        self._header = msg.header

    def convert_depth_image(self, image_msg):
        if image_msg is not None:
            depth_image = self.br.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            return depth_array
        else:
            return None

    def get_img(self):
        result = self.image
        return result

    def get_depth_array(self):
        result = self.depth_array
        return result

    def getResult(self, predictions, classes):

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            #print(type(masks))
        else:
            return

        result_msg = Result()
        result_msg.header = self._header
        result_msg.class_ids = predictions.pred_classes if predictions.has("pred_classes") else None
        result_msg.class_names = np.array(classes)[result_msg.class_ids.numpy()]
        result_msg.scores = predictions.scores if predictions.has("scores") else None

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]]=255
            mask = self.br.cv2_to_imgmsg(mask)
            result_msg.masks.append(mask)

            box = RegionOfInterest()
            box.x_offset = np.uint32(x1)
            box.y_offset = np.uint32(y1)
            box.height = np.uint32(y2 - y1)
            box.width = np.uint32(x2 - x1)
            result_msg.boxes.append(box)

        return result_msg

    def sort_detections(self, msg):
        # Sort detections by y-position of upper left bbox corner
        # TODO: Sort by y-position first and then sort again by x-position
        # This will prevent object ids from flipping back and forth if they are at the same y-position

        target = self.Out_transfer(msg.class_ids, msg.class_names, msg.scores, msg.boxes, msg.masks)

        # Sort by y-offset
        # self.Sort_quick(target, 0, len(target)-1)

        # Sort by y-offset and x-offset
        self.Sort_quick(target, 0, len(target)-1, y=True)

        #after sort the y, then start sorting the x:
        arr_y = [(target[w][3].y_offset + target[w][3].y_offset + target[w][3].height)/2 for w in range(len(target))] #(y1+y2)/2

        store = []
        for i in range(len(arr_y)):
            if arr_y.count(arr_y[i]) > 1:
                store.append([i, arr_y.count(arr_y[i])+1])

        if len(store) !=0:
            for each_group in store:
                self.Sort_quick(target, each_group[0], each_group[1], y=False)

        return target

    def Out_transfer(self, class_id, class_name, score, box, mask):

        num = int(len(class_id))
        target = []

        for i in range(num):

            target.append([class_id[i], class_name[i], score[i], box[i], mask[i]])

        return target

    def partition(self, target, low, high, y=True):

        i = ( low-1 )
        arr = []
        # pdb.set_trace()
        if y:
            # x1 = target[w][3].x_offset
            # y1 = target[w][3].y_offset
            # x2 = target[w][3].x_offset + target[w][3].width
            # y2 = target[w][3].y_offset + target[w][3].height
            arr = [(target[w][3].y_offset + target[w][3].y_offset + target[w][3].height)/2 for w in range(len(target))] #box:[x1, y1, x2, y2]  value :(y1+y2)/2
        else:
            arr = [(target[w][3].x_offset + target[w][3].x_offset + target[w][3].width)/2 for w in range(len(target))] #box:[x1, y1, x2, y2]  value :(x1+x2)/2

        pivot = arr[high]

        for j in range(low , high): 
            if   arr[j] <= pivot: 
                i = i+1 
                target[i],target[j] = target[j],target[i] 
    
        target[i+1],target[high] = target[high],target[i+1] 

        return ( i+1 )

    def Sort_quick(self, target, low, high, y):

        if low < high: 
            pi = self.partition(target,low,high, y) 
    
            self.Sort_quick(target, low, pi-1, y) 
            self.Sort_quick(target, pi+1, high, y)

    def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None

        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        return cv_img

    def camInfo_callback(self, msg):
        self.cam_info = msg

    def convert_depth_to_phys_coord_using_realsense(self, x, y, depth, cameraInfo):
        _intrinsics = pyrealsense2.intrinsics()
        _intrinsics.width = cameraInfo.width
        _intrinsics.height = cameraInfo.height
        _intrinsics.ppx = cameraInfo.K[2]
        _intrinsics.ppy = cameraInfo.K[5]
        _intrinsics.fx = cameraInfo.K[0]
        _intrinsics.fy = cameraInfo.K[4]
        #_intrinsics.model = cameraInfo.distortion_model
        _intrinsics.model  = pyrealsense2.distortion.none
        _intrinsics.coeffs = [i for i in cameraInfo.D]
        result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
        # result[0]: right, result[1]: down, result[2]: forward
        return result[0], result[1], result[2]
        # return result[2], -result[0], -result[1]

    def publish(self, img, result_msg, cloud_samples_msg):
        self.pub.publish(img)
        self.result_pub.publish(result_msg)
        self.pc_pub.publish(cloud_samples_msg)
        self.loop_rate.sleep()


def main():
    """ Mask RCNN Object Detection with Detectron2 """
    rospy.init_node("mask_rcnn", anonymous=True)
    bridge = CvBridge()
    start_time = time.time()
    image_counter = 0
    
    register_coco_instances("train_set", {}, "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/train/annotations.json", "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/train")
    register_coco_instances("test_set", {}, "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/test/annotations.json", "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/test")
    
    train_metadata = MetadataCatalog.get("train_set")
    print(train_metadata)
    dataset_dicts_train = DatasetCatalog.get("train_set")

    test_metadata = MetadataCatalog.get("test_set")
    print(test_metadata)
    dataset_dicts_test = DatasetCatalog.get("test_set")

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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("test_set")
    predictor = DefaultPredictor(cfg)

    class_names = MetadataCatalog.get("train_set").thing_classes

    # Set up custom cv2 visualization parameters
    # Classes: [name, id]
    #               -
    #          [Plate,   0]
    #          [Carrot,  1]
    #          [Celery,  2]
    #          [Pretzel, 3]
    #          [Gripper, 4]

    # Colors = [blue, green, red]
    color_plate = [0, 255, 0]       # green
    color_carrot = [255, 200, 0]    # blue
    color_celery = [0, 0, 255]      # red
    color_pretzel = [0, 220, 255]   # yellow
    color_gripper = [204, 0, 150]   # purple
    colors = list([color_plate, color_carrot, color_celery, color_pretzel, color_gripper])

    alpha = .4

    run = maskRCNN()
    while not rospy.is_shutdown():
        # Get images
        img = run.get_img()

        if img is not None:
            outputs = predictor(img)
            predictions = outputs["instances"].to("cpu")

            # Get results
            unsorted = run.getResult(predictions, class_names)

            # Sort detections by x and y
            sorted = run.sort_detections(unsorted)

            result = Result()
            for i in range(len(sorted)):
                result.class_ids.append(sorted[i][0])
                result.class_names.append(sorted[i][1])
                result.scores.append(sorted[i][2])
                result.boxes.append(sorted[i][3])
                result.masks.append(sorted[i][4])

            # Visualize using detectron2 built in visualizer
            # v = Visualizer(im[:, :, ::-1],
            #             metadata=train_metadata, 
            #             scale=1.0 
            #             # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            # )
            # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # im = v.get_image()[:, :, ::-1]
            # im_msg = bridge.cv2_to_imgmsg(im, encoding="bgr8")

            # Visualize using custom cv2 code
            if result is not None:
                result_cls = result.class_names
                result_clsId = result.class_ids
                result_scores = result.scores
                result_masks = result.masks

                # Create copies of the original image
                im = img.copy()
                output = img.copy()

                # Initialize lists
                masks = []
                masks_indices = []
                for i in range(len(result_clsId)):
                    # Obtain current object mask as a numpy array (black and white mask of single object)
                    current_mask = bridge.imgmsg_to_cv2(result_masks[i])

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

                im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")

                ##### The entire goal of the below code is to get N random points on the mask in 3D
                ##### and publish on cloud samples topic for GPD
                item_ids = result_clsId
                idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]
                numFoodItems = len(idx)

                mask = bridge.imgmsg_to_cv2(result_masks[idx[0]])
                coord = cv2.findNonZero(mask)   # Coordinates of the mask that are on the food item

                # Pick 3 random points on the object mask
                sample_list = list()
                for ii in range(3):
                    point = Point()
                    x = random.choice(coord[:,0,1]) # x and y reversed for some reason
                    y = random.choice(coord[:,0,0]) # x and y reversed for some reason
                    depth = (run.depth_array[y, x]) / 1000
                    # Deproject pixels and depth to 3D coordinates (camera frame)
                    X, Y, Z = run.convert_depth_to_phys_coord_using_realsense(y, x, depth, run.cam_info)
                    # print("(x,y,z) to convert: ("+str(y)+", "+str(x)+", "+str(depth)+")")
                    # print("(X,Y,Z) converted: ("+str(X)+", "+str(Y)+", "+str(Z)+")")
                    point.x = X; point.y = Y; point.z = Z
                    sample_list.append(point)

                # print(sample_list)

                cam_source = Int64()
                cam_source.data = 0

                cloud_source = CloudSources()
                cloud_source.cloud = run.pointCloud
                cloud_source.camera_source = [cam_source]
                view_point = Point()
                view_point.x = 0.640; view_point.y = 0.828; view_point.z = 0.505
                # view_point.x = 0; view_point.y = 0; view_point.z = 0
                cloud_source.view_points = [view_point]

                cloud_samples = CloudSamples()
                cloud_samples.cloud_sources = cloud_source
                cloud_samples.samples = sample_list

                # Print publish info
                # print(type(cloud_source.cloud))
                # print(cloud_source.camera_source)
                # print(cloud_source.view_points)
                # print("")
                # print(type(cloud_samples.cloud_sources))
                # print(cloud_samples.samples)
                # print("-------------------------\n")

            # Display Image Counter
            # image_counter = image_counter + 1
            # if (image_counter % 11) == 10:
            #     rospy.loginfo("Images detected per second=%.2f", float(image_counter) / (time.time() - start_time))

            run.publish(im_msg, result, cloud_samples)    
        

    return 0

if __name__ == '__main__':
    sys.exit(main())
import pyrealsense2 as rs
import numpy as np
import cv2
import json

#%%
DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C", "0B64"]

def find_device_json_input_interface() :
    ctx = rs.context()
    ds5_dev = rs.device()
    dev = ctx.query_devices()
    return dev
    # devices = ctx.query_devices()
    # for dev in devices:
    #     if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
    #         if dev.supports(rs.camera_info.name):
    #             print("Found device", dev.get_info(rs.camera_info.name))
    #         return dev
    # raise Exception("No product line device that has json input interface")

jsonDict = json.load(open("/home/labuser/L515 settings.json"))
jsonString= str(jsonDict).replace("'", '\"')


#%%
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

try:
    dev = find_device_json_input_interface()
    ser_dev = rs.serializable_device(dev)  
    ser_dev.load_json(jsonString)
    print("loaded json")


except Exception as e:
    print(e)
    pass
#%%


# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

#後処理フィルタブロック定義
decimation = rs.decimation_filter()
# decimation.set_option(rs.option.filter_magnitude, 4)

depth_to_disparity = rs.disparity_transform(False)


spatial = rs.spatial_filter()
# spatial.set_option(rs.option.filter_magnitude, 5)
# spatial.set_option(rs.option.filter_smooth_alpha, 1)
# spatial.set_option(rs.option.filter_smooth_delta,50)

temporal = rs.temporal_filter()
#temporal.set_option(rs.option.filter_magnitude, 5)
temporal.set_option(rs.option.filter_smooth_alpha, 0.1)
temporal.set_option(rs.option.filter_smooth_delta,100)

disparity_to_depth = rs.disparity_transform(False)

hole_filling = rs.hole_filling_filter()
# spatial.set_option(rs.option.holes_fill, 3)

colorizer = rs.colorizer()

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        #後処理
        # Postframe3 = depth_frame
        
        # Postframe1 = decimation.process(depth_frame)
        # Postframe2 = depth_to_disparity.process(Postframe1)
        # Postframe3 = spatial.process(Postframe2)
        # Postframe4 = temporal.process(Postframe3)
        # Postframe5 = disparity_to_depth.process(Postframe4)
        # Postframe6 = hole_filling.process(Postframe5)
        
        Postframe6 = temporal.process(depth_frame)
        
        # Postframe6 = Postframe4
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        #depth_image = np.asanyarray(depth_frame.get_data())
        
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())
        Postframe_colormap = np.asanyarray(colorizer.colorize(Postframe6).get_data())
        
        ##The code for this sample uses applycolormap instead of colorizer
        # Postframe_image = np.asanyarray(Postframe6.get_data())
        
# =============================================================================
#         # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#         ## Colormap applied for filter as well
#         Postframe_colormap = cv2.applyColorMap(cv2.convertScaleAbs(Postframe_image, alpha=0.03), cv2.COLORMAP_JET)
# =============================================================================
        

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        Postframe_colormap_dim = Postframe_colormap.shape

# =============================================================================
#         #If depth and color resolutions are different, resize color image to match depth image for display
#         if depth_colormap_dim != color_colormap_dim:
#             resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
#             images = np.hstack((resized_color_image, depth_colormap))
#         else:
#             images = np.hstack((color_image, depth_colormap))
#         
# 
# =============================================================================

        if depth_colormap_dim != Postframe_colormap_dim:
            resized_Postframe_image = cv2.resize(Postframe_colormap, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_Postframe_image, depth_colormap))
        else:
            images = np.hstack((Postframe_colormap, depth_colormap))            

            
            
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

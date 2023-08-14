import os.path

import requests
import json
import cv2
import numpy as np
import math
def generate_mask(image_size, grid_size, prob_thresh):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
            prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask

def iou(box1, box2):
    x1 = max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
    y1 = max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
    x2 = min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)
    y2 = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    b_area = box2[2] * box2[3]
    return intersection_area / float(box1[2]*box1[3] + b_area - intersection_area)

def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
              255).astype(np.uint8)
    return masked

def transform_masked(image):
    input_size = (640, 640)  # 输入图像的大小
    image = cv2.resize(image, input_size)
    image = image.transpose(2, 0, 1)  # 转置通道顺序为CHW
    image = np.expand_dims(image, axis=0)  # 添加批次维度
    image = image / 255.0  # 归一化到 [0, 1]
    return image

def get_masked_outputs(masked,api_url,iou_threshold,confidence_threshold):
    print(masked.shape)
    # 构造请求的 JSON 数据
    request_data = {"data": masked.tolist(),"iou_threshold":iou_threshold,"confidence_threshold":confidence_threshold}  # 注意要转换为列表形式

    # 发送 POST 请求到 API
    response = requests.post(api_url, json=request_data)
    # print('11111111111111111111')
    # 解析响应的 JSON 数据
    result = response.json()

    # 输出推断结果
    # print("Prediction maksed result:", result["result"])
    return np.array(result['result'])

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    print('gen_cam_mask',mask)
    mask = norm_image(mask)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    cam = 0.5 * heatmap + 0.5 * image
    return norm_image(cam), heatmap

def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def transfer_box(box,type,image_with_bbox):
    if(type=='xyxy'):
        return box
    else:
        image_shape = image_with_bbox.shape
        # print(image_shape)
        weight = image_shape[1]
        height = image_shape[0]
        # print(weight,height)
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        x1 = x1/640*weight
        y1 = y1/640*height
        x2 = x2 / 640 * weight
        y2 = y2 / 640 * height
        # print(x1,y1,x2,y2)
        box[0]=x1 - int(x2 / 2)
        box[1]=y1 - int(y2 / 2)
        box[2]=x1 + int(x2 / 2)
        box[3]=y1 + int(y2 / 2)
    return box

def generate_saliency_map(image,api_url,
                          target_class_index,#检测框的类的索引下标
                          target_box,#当前检测框
                          prob_thresh=0.5,
                          grid_size=(16, 16),
                          n_masks=2000,
                          seed=0):
    np.random.seed(seed)
    # print(target_class_index)
    image_h,image_w = image.shape[:2]
    res = np.zeros((image_h,image_w),dtype=np.float32)
    masked_iou_threshold=0.3
    masked_confidence_threshold=0.4
    for _ in range(n_masks):
        mask = generate_mask(image_size=(image_w, image_h),
                             grid_size=grid_size,
                             prob_thresh=prob_thresh)

        masked = mask_image(image, mask)
        masked = transform_masked(masked)
        # print("first masked",masked)
        masked_pred_result = get_masked_outputs(masked,api_url,masked_iou_threshold,masked_confidence_threshold)
        # print('masked_pred_result',masked_pred_result,len(masked_pred_result))
        for i in range(len(masked_pred_result)):
            # print('target_class_index',target_class_index[0])
            # print('masked_pred_result',masked_pred_result[i][5:6])
            if masked_pred_result[i][5:6] == target_class_index[0]:
                masked_filtered_result_box = masked_pred_result[i][0:4]
                masked_filtered_result_score = masked_pred_result[i][4:5]
                print(iou(target_box, masked_filtered_result_box))
                score = max([iou(target_box, masked_filtered_result_box) * masked_filtered_result_score],
                            default=0)
                res += mask * score
    return res

def generate_heatmap(pred_result_array,image,api_url,save_dir):
    for i in range(len(pred_result_array)):
        result_box=pred_result_array[i][0:4]
        result_class_index = pred_result_array[i][5:6]
        # print(result_class_index)
        saliency_map = generate_saliency_map(image,api_url,
                                             result_class_index,  # 检测框的类的索引下标
                                             result_box,  # 当前检测框
                                             prob_thresh=0.5,
                                             grid_size=(16, 16),
                                             n_masks=2000,
                                             seed=0)
        print('saliency_map',saliency_map)
        image_with_bbox, heatmap = gen_cam(image, saliency_map)
        result_new_box = transfer_box(result_box, 'xywh', image_with_bbox)

        cv2.rectangle(image_with_bbox, (int(result_new_box[0]), int(result_new_box[1])),
                      (int(result_new_box[2]), int(result_new_box[3])), (0, 0, 255), 5)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir,'dyj_test_' + label_names[int(result_class_index[0])] + '_' + str(i) + ".jpg")

        cv2.imwrite(save_path, image_with_bbox)
    return 0

label_names = [
    "Human", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Ship", "Traffic Light",
    "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow",
    "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Other Luggage",
    "TV", "Monitor", "Laptop", "Mouse", "Cell Phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator",
    "Book", "Clock", "Vase", "Scissors", "Panda", "Television", "Dining Table", "Chair", "Mirror", "Cutlery",
    "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut",
    "Cake", "Armchair", "Sofa", "Potted Plant", "Bed", "Dining Table", "Toilet", "Television Studio", "Book",
    "Front Door","Back Door", "Window", "Cabinet", "Picture", "Bathtub", "Towel", "Pillow", "Screen", "Tablecloth", "Desk Lamp"
]

# 定义 API 地址
api_url = "http://localhost:5000/predict"

# 准备输入图片（替换为你的图片路径）
image_path = "/data/zrj/zhihangyuan/onnx_test_data/000000000009.jpg"

image = cv2.imread(image_path)
generate_heatmap_image = image.copy()
# weight = image.shape[1]
# height = image.shape[0]
# print(weight,height)
input_size = (640, 640)  # 输入图像的大小
image = cv2.resize(image, input_size)
image = image.transpose(2, 0, 1)  # 转置通道顺序为CHW
image = np.expand_dims(image, axis=0)  # 添加批次维度
image = image / 255.0  # 归一化到 [0, 1]
# print(image.shape)
iou_threshold = 0.5
confidence_threshold = 0.4
# 构造请求的 JSON 数据
request_data = {"data": image.tolist(),"iou_threshold":iou_threshold,"confidence_threshold":confidence_threshold}  # 注意要转换为列表形式

# 发送 POST 请求到 API
response = requests.post(api_url, json=request_data)

# 解析响应的 JSON 数据
result = response.json()

# 输出推断结果
pred_result_array=np.array(result['result'])
print("Prediction result:", pred_result_array,type(pred_result_array),pred_result_array.shape)
save_dir = '/data/zrj/zhihangyuan/YOLO5/yolov5/test_images'
print('start generate heatmap')
generate_heatmap(pred_result_array,generate_heatmap_image,api_url,save_dir)
print('finish generate')


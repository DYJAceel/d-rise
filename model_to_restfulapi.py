import torch
from models.yolo import Model
import cv2
import numpy as np
import onnxruntime
import os
from flask import Flask, request, jsonify

def calculate_iou(box, boxes):
    """
    计算目标框与剩余框列表中每个框之间的IoU值
    box: 目标框的坐标，格式为 [xmin, ymin, xmax, ymax]
    boxes: 剩余框列表，每个框的坐标格式为 [xmin, ymin, xmax, ymax]
    返回IoU值的列表
    """
    # 计算目标框的面积
    box_area = box[2] * box[3]

    ious = []
    for b in boxes:
        # 计算相交区域的坐标
        x1 = max(box[0]-box[2]/2, b[0]-b[2]/2)
        y1 = max(box[1]-box[3]/2, b[1]-b[3]/2)
        x2 = min(box[0]+box[2]/2, b[0]+b[2]/2)
        y2 = min(box[1]+box[3]/2, b[1]+b[3]/2)

        # 计算相交区域的面积
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # 计算剩余框的面积
        b_area = b[2] * b[3]

        # 计算IoU值
        iou = intersection_area / float(box_area + b_area - intersection_area)

        ious.append(iou)

    ious = torch.tensor(ious)

    return ious
def count_unique_values(tensor):
    """
    统计 Tensor 中不同数的个数，并返回从小到大排序的列表
    tensor: 输入的 Tensor
    返回从小到大排序的列表
    """
    unique_values = torch.unique(tensor)  # 获取 Tensor 中的唯一值
    sorted_values = torch.sort(unique_values).values  # 对唯一值进行排序
    unique_count = sorted_values.size(0)  # 统计唯一值的个数

    return sorted_values.tolist()
def filter_boxes(detections, confidence_threshold, iou_threshold):
    # 从detections中提取边界框的相关信息
    boxes = detections[:, :4]  # 边界框坐标
    scores = detections[:, 4]  # 置信度得分
    labels = detections[:, 5:]  # 对象标签


    labels = torch.argmax(labels, dim=1)

    detections = []
    # 置信度阈值筛选
    mask = scores >= confidence_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    list = count_unique_values(filtered_labels)

    print("非极大抑制前")
    print(filtered_boxes)
    print(filtered_scores)
    print(filtered_labels)

    for i in range(len(list)):

        mask = filtered_labels == list[i]
        filtered_boxess = filtered_boxes[mask]
        filtered_scoress = filtered_scores[mask]
        filtered_labelss = filtered_labels[mask]

        flag = 0
        # 非最大抑制
        while True:

            flag+=1
            ori_shape = filtered_labelss.shape

            if flag > filtered_labelss.shape[0]:
                break

            values, indices = torch.topk(filtered_scoress, flag)

            # # 选择得分最高的边界框
            # max_idx = filtered_scoress.argmax()
            max_idx = indices[flag-1]


            # 计算与其他边界框的IoU
            ious = calculate_iou(filtered_boxess[max_idx], filtered_boxess)
            # print(ious)

            # 筛选掉与当前边界框重叠程度高于阈值的边界框
            mask = ious <= iou_threshold
            # print(mask)

            mask[max_idx] = True

            filtered_boxess = filtered_boxess[mask]
            filtered_scoress = filtered_scoress[mask]
            filtered_labelss = filtered_labelss[mask]

            if filtered_labelss.shape == ori_shape:
                break

        # 返回筛选后的边界框、置信度得分和对象标签
        filtered_detections = torch.cat([
            filtered_boxess,
            filtered_scoress.unsqueeze(1),
            filtered_labelss.unsqueeze(1)
        ], dim=1)


        for i in filtered_detections.tolist():
            detections.append(i)


    return detections

app = Flask(__name__)

# 加载你的 ONNX 模型（这里以一个简单的示例模型为例）
onnx_model_path = 'xxxx/xxxx/xxxx'  # 替换为你的模型路径
session = onnxruntime.InferenceSession(onnx_model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # step1 从请求的 JSON 数据中获取输入数据
        data = request.json['data']

        iou_threshold = request.json['iou_threshold']

        confidence_threshold = request.json['confidence_threshold']

        # step2 数据预处理
        # 将输入数据转换为 NumPy 数组并设置数据类型
        input_data = np.array(data, dtype=np.float32)

        # step3 使用 ONNX 运行会话来进行推断
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: input_data})[0][0]

        # step4 数据后处理
        result = torch.from_numpy(result)
        results = filter_boxes(result,confidence_threshold,iou_threshold)

        # step5 返回推断结果作为 JSON 响应
        return jsonify({"result": results})

    except Exception as e:
        # 处理异常情况并返回错误信息
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    # 在本地启动 Flask 应用，监听在指定的主机和端口
    app.run(host='0.0.0.0', port=5000)

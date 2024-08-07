import numpy as np
import cv2
import random
from scipy.ndimage import rotate

def get_random_pairs(num_pairs, patch_size):
    """
        生成随机比较对
        :param num_pairs BRIEF描述子点对个数
        :param patch_size 取点的范围（正方形边长）
    """
    pairs = []
    half_patch = patch_size // 2
    for _ in range(num_pairs):
        x1, y1 = np.random.randint(-half_patch, half_patch, size=2)
        x2, y2 = np.random.randint(-half_patch, half_patch, size=2)
        pairs.append(((x1, y1), (x2, y2)))
    return pairs

def generate_brief_descriptors(keypoints, image, pairs, patch_size):
    """
        生成 BRIEF 描述子
        :param pairs: 随机点对
        :param keypoints: 特征点列表
        :param image: 输入图像（灰度图）
        :param patch_size: 特征点周围区域的大小
        :return: 描述子数组
    """
    def extract_patch(image, keypoint, patch_size):
        """提取特征点周围的补丁"""
        x, y = int(keypoint[0]), int(keypoint[1])
        half_patch = patch_size // 2
        x0, y0 = max(x - half_patch, 0), max(y - half_patch, 0)
        x1, y1 = min(x + half_patch, image.shape[0]), min(y + half_patch, image.shape[1])
        patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
        patch[x0 - (x - half_patch):x1 - (x - half_patch), y0 - (y - half_patch):y1 - (y - half_patch)] = image[x0:x1, y0:y1]
        return patch

    def compute_descriptor(patch, pairs):
        """计算 BRIEF 描述子"""
        descriptor = []
        # patch = rotate(patch, -orientation, reshape=False)
        for (p1, p2) in pairs:
            try:
                if patch[p1[1] + patch_size//2, p1[0] + patch_size//2] < patch[p2[1] + patch_size//2, p2[0] + patch_size//2]:
                    descriptor.append(1)
                else:
                    descriptor.append(0)
            except IndexError:
                descriptor.append(0)
        # return int(descriptor, 2)
        return descriptor

    descriptors = []
    for kp in keypoints:
        # 提取特征点周围的补丁
        patch = extract_patch(image, kp, patch_size)
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            continue
        descriptor = compute_descriptor(patch, pairs)
        descriptors.append(descriptor)

    return np.array(descriptors)

def match(img1, img2, shape, kp1, kp2, descriptors1, descriptors2, count, mode):
    """
        两图特征点进行匹配
        :param img1: 图像1
        :param img2: 图像2
        :param shape: 图像尺寸
        :param kp1: 图像1特征点坐标
        :param kp2: 图像2特征点坐标
        :param descriptors1: 图像1的BRIEF描述子
        :param descriptors2: 图像2的BRIEF描述子
        :param count: 所需匹配对个数
        :param mode: 匹配模式 1：按匹配度排序输出前若干个；2：随机选若干个
        :return: 描述子数组
    """
    mt = []
    for i, descriptor1 in enumerate(descriptors1):
        minv = 1024
        flag = -1
        for j, descriptor2 in enumerate(descriptors2):
            score = sum(np.abs(np.array(descriptor1) - np.array(descriptor2)))
            if score < minv:
                minv = score
                flag = j
        mt.append((i, flag, minv))

    img = cv2.hconcat([img1, img2])
    slope = []

    for i, match in enumerate(mt):
        start_point = (kp1[match[0]][1], kp1[match[0]][0])
        end_point = (kp2[match[1]][1] + shape[0], kp2[match[1]][0])
        slope.append((start_point[1] - end_point[1]) / (start_point[0] - end_point[0]))

    mean = sum(slope) / len(slope)
    variance = sum((x - mean) ** 2 for x in slope) / (len(slope) - 1)

    if mode==1:# 按汉明距离从小到大顺序查找
        mt = sorted(mt, key=lambda x: x[2])
    elif mode==2:# 随机挑选
        mt = random.sample(mt, len(mt))

    print(len(mt))

    for i, match in enumerate(mt):
        if i==count:
            break
        start_point = (kp1[match[0]][1], kp1[match[0]][0])
        end_point = (kp2[match[1]][1] + shape[0], kp2[match[1]][0])

        if mean - 2 * variance < (start_point[1] - end_point[1]) / (start_point[0] - end_point[0]) < mean + 2 * variance:
            cv2.line(img, start_point, end_point, random.randint(0, 255), 1)

    return img
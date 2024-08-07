import cv2
import numpy as np
from PIL import Image,ImageDraw
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

def get_pixel_value(img, coords): #读取非整点的像素值
    """
        读取非整点的像素值
        :param img: 输入图像;
        :param coords:非整数点坐标
        :return: 非整数点像素值
    """
    x, y = coords
    map_x = np.float32([x])
    map_y = np.float32([y])

    pixel_value = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return pixel_value[0, 0]

def method3_main_harries_getdata(img, T, r, angle_deg):
    """
        根据harries方法获得特征点与非特征点数据
        :param img: 图片
        :param T: 图片裁剪的边缘比例
        :param r: 获取特征点周围像素值的半径
        :param angle_deg: 周围方向均分的角度（角度制）
        :return: train_data和train_lab numpy类型
    """
    # gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # gray = cv2.resize(gray, shape)
    # gray = adjust_brightness(gray, bright)
    # if sharpen==1:
    #     gray = sharpen_image(gray)
    # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = img
    gray_arr = np.array(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.08)
    dst = cv2.dilate(dst, None)

    corners = np.argwhere(dst > 0.01 * dst.max())
    corners = [tuple(c) for c in corners]

    vis=np.array(gray)
    H=gray.shape[0]
    W=gray.shape[1]
    angle_rad = np.deg2rad(angle_deg)
    train_data = []
    train_lab = []
    for i, corner in enumerate(corners):
        train_data_ori = []
        vis[corner[0],corner[1]]=1
        if corner[0] < T * H or corner[0] > H - T * H:
            continue
        if corner[1] < T * W or corner[1] > W - T * W:
            continue
        for j in range(np.int64(360 // angle_deg)):
            x = corner[0] + r * np.sin(j * angle_rad)
            y = corner[1] + r * np.cos(j * angle_rad)
            train_data_ori.append(
                # np.float32(get_pixel_value(gray, [y, x])) - np.float32(get_pixel_value(gray, [corner[1], corner[0]])))
                np.float32(gray_arr[round(x), round(y)]) - np.float32(gray_arr[corner[0], corner[1]]) )
        # cv2.circle(gray, (corner[1],corner[0]), radius=1, color=133, thickness=-1)
        train_data.append(train_data_ori)
        train_lab.append(1)

    perc = len(train_lab)*10 / (H * W * (1-2*T)**2)
    print(perc)

    for x in range(int(T * H),int(H - T * H)):
        for y in range(int(T * W), int(W - T * W)):
            if vis[x,y] != 1:
                if random.random() < perc:
                    train_data_ori = []
                    for j in range(np.int64(360 // angle_deg)):
                        xx = x + r * np.sin(j * angle_rad)
                        yy = y + r * np.cos(j * angle_rad)
                        train_data_ori.append(
                            # np.float32(get_pixel_value(gray, [yy, xx])) - np.float32(get_pixel_value(gray, [y, x])))
                            np.float32(gray_arr[round(xx), round(yy)]) - np.float32(gray_arr[x, y]) )
                    # cv2.circle(gray, (y,x), radius=1, color=1, thickness=-1)
                    train_data.append(train_data_ori)
                    train_lab.append(0)

    return np.array(train_data), np.array(train_lab)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

def train_Teacher(img, T, r, angle_deg, ModelType, ModelParam, batch_size, epochs, device):
    """
        教师模型的训练
        :param img: 图片
        :param T: 图片裁剪的边缘比例
        :param r: 获取特征点周围像素值的半径
        :param angle_deg: 周围方向均分的角度（角度制）
        :param ModelParam: 各模型的参数元组
        :param ModelType: 模型代号 1:SimpleRNN   2:Transformer
        :param batch_size: batch大小
        :param epochs: 训练循环次数
        :param device: 设备
        :return: 无，但会保存Teacher模型为Teacher.pth
    """
    data,lab = method3_main_harries_getdata(img, T, r, angle_deg)
    data = torch.from_numpy(data)
    lab = torch.from_numpy(lab).view(-1, 1)

    dataset = TensorDataset(data, lab)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Number of batches: {len(train_loader)}")

    input_size = int(360//angle_deg)
    output_size = 1
    model = 0
    if ModelType==1:
        model = SimpleRNN(int(input_size), int(ModelParam[0]), int(output_size))
    elif ModelType==2:
        model = TransformerClassifier(int(input_size), ModelParam[0], ModelParam[1], ModelParam[2], 1,
                                      ModelParam[3])
    model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 将输入和标签移动到设备（如果有GPU，则移动到GPU）
            # print(labels.shape)
            inputs = inputs.view(-1, 1, input_size)
            labels = labels.float()
            outputs = model(inputs).to(device)
            # print(outputs.shape)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.view(-1, 1, input_size)
            labels = labels.float()
            outputs = model(inputs).to(device)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy on training data: {100 * correct / total:.2f}%')

    torch.save(model.state_dict(), 'Teacher.pth')

def train_Student(img, T, r, angle_deg, TeacherModelType, ModelParamTeacher, StudentModelType, ModelParamStudent,
                               batch_size, epochs, temperature, alpha, device):
    """
        学生模型的训练，即知识蒸馏
        :param img: 图片
        :param T: 图片裁剪的边缘比例
        :param r: 获取特征点周围像素值的半径
        :param angle_deg: 周围方向均分的角度（角度制）
        :param TeacherModelType: 教师模型代号 1:SimpleRNN   2:Transformer
        :param ModelParamTeacher: 教师模型的参数
        :param StudentModelType: 学生模型代号 1:SimpleRNN   2:Transformer
        :param ModelParamStudent: 学生模型的参数
        :param batch_size: batch大小
        :param epochs: 训练循环次数
        :param temperature: 知识蒸馏温度参数
        :param alpha: 知识蒸馏alpha参数
        :param device 设备
        :return: 无，但会保存Student模型为Student.pth
    """
    # 创建教师和学生模型
    input_size = int(360 // angle_deg)
    output_size = 1
    student_model = 0
    teacher_model = 0
    if TeacherModelType == 1:
        teacher_model = SimpleRNN(input_size, int(ModelParamTeacher[0]), int(output_size))
    elif TeacherModelType == 2:
        teacher_model = TransformerClassifier(int(input_size), ModelParamTeacher[0], ModelParamTeacher[1],
                                              ModelParamTeacher[2], 1, ModelParamTeacher[3])
    if StudentModelType == 1:
        student_model = SimpleRNN(input_size, int(ModelParamStudent[0]), int(output_size))
    elif StudentModelType == 2:
        student_model = TransformerClassifier(int(input_size), ModelParamStudent[0], ModelParamStudent[1],
                                              ModelParamStudent[2], 1, ModelParamStudent[3])

    student_model.to(device)
    teacher_model.to(device)

    teacher_model.load_state_dict(torch.load('Teacher.pth'))
    teacher_model.eval()

    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss().to(device)

    data, lab = method3_main_harries_getdata(img, T, r, angle_deg)
    data = torch.from_numpy(data)
    lab = torch.from_numpy(lab).view(-1, 1)

    dataset = TensorDataset(data, lab)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        student_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.view(-1, 1, input_size)
            labels = labels.float()

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            # 前向传播学生模型
            student_outputs = student_model(inputs)

            # 知识蒸馏损失
            loss_soft = F.kl_div(F.log_softmax(student_outputs / temperature, dim=1),
                                 F.softmax(teacher_outputs / temperature, dim=1), reduction='batchmean') * (
                                    temperature * temperature)
            loss_hard = criterion(student_outputs, labels)
            loss = alpha * loss_hard + (1 - alpha) * loss_soft

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    student_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.view(-1, 1, input_size)
            labels = labels.float()
            outputs = student_model(inputs).to(device)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy on training data: {100 * correct / total:.2f}%')

    torch.save(student_model.state_dict(), 'Student.pth')

def compute_centroid(image_patch):
    # 计算质心
    M, N = image_patch.shape
    x_coords, y_coords = np.meshgrid(np.arange(N), np.arange(M))
    total_intensity = np.sum(image_patch)
    x_centroid = np.sum(x_coords * image_patch) / total_intensity
    y_centroid = np.sum(y_coords * image_patch) / total_intensity
    return x_centroid, y_centroid

def compute_orientation(image_patch, x_centroid, y_centroid):
    # 计算方向
    M, N = image_patch.shape
    x_coords, y_coords = np.meshgrid(np.arange(N), np.arange(M))
    x_diff = x_coords - x_centroid
    y_diff = y_coords - y_centroid
    orientation = np.arctan2(np.sum(y_diff * image_patch), np.sum(x_diff * image_patch))
    return np.degrees(orientation)

def Predict(img, T, r, angle_deg, ModelType, ModelParam, model_path, t,  device):
    """
        使用特定模型检测特征点
        :param img: 图片
        :param T: 图片裁剪的边缘比例
        :param r: 获取特征点周围像素值的半径
        :param angle_deg: 周围方向均分的角度（角度制）
        :param ModelParam: 各模型的参数元组
        :param ModelType: 模型代号 1:SimpleRNN   2:Transformer
        :param model_path: 使用模型的名称
        :param t sigmoid阈值
        :param device: 设备
        :return: kp列表表示特征点坐标
    """
    input_size = int(360 // angle_deg)
    output_size = 1
    device = torch.device(device)
    if ModelType==1:
        model = SimpleRNN(input_size, int(ModelParam[0]), int(output_size))
    else:
        model = TransformerClassifier(int(input_size), ModelParam[0], ModelParam[1],
                                              ModelParam[2], 1, ModelParam[3])
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    # gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # gray = cv2.resize(gray, shape)
    # gray = adjust_brightness(gray, bright)
    # if sharpen == 1:
    #     gray = sharpen_image(gray)
    gray = img
    H=gray.shape[0]
    W=gray.shape[1]
    gray_arr = np.array(gray)
    gray_out = gray
    angle_rad=np.deg2rad(angle_deg)
    kp = []
    patch_size = 7

    with torch.no_grad():
        for x in range(int(T * H), int(H - T * H), r):
            for y in range(int(T * W), int(W - T * W), r):
                data_ori = []
                for j in range(np.int64(360 // angle_deg)):
                    xx = x + r * np.sin(j * angle_rad)
                    yy = y + r * np.cos(j * angle_rad)
                    data_ori.append(
                        #np.float32(get_pixel_value(gray, [yy, xx])) - np.float32(get_pixel_value(gray, [y, x])))
                        np.float32(gray_arr[round(xx), round(yy)]) - np.float32(gray_arr[x, y]) )
                data = torch.from_numpy(np.array(data_ori)).unsqueeze(0).unsqueeze(0).to(device)
                #print(data)
                output = model(data)
                if output > t:
                    # cv2.circle(gray_out, (y, x), radius=1, color=10, thickness=-1)
                    # half_patch = patch_size // 2
                    # x0, y0 = max(x - half_patch, 0), max(y - half_patch, 0)
                    # x1, y1 = min(x + half_patch, gray.shape[0]), min(y + half_patch, gray.shape[1])
                    # patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
                    # patch[x0 - (x - half_patch):x1 - (x - half_patch), y0 - (y - half_patch):y1 - (y - half_patch)] \
                    #     = gray[x0:x1, y0:y1]
                    #
                    # x_centroid, y_centroid = compute_centroid(patch)
                    # orientation = compute_orientation(patch, x_centroid, y_centroid)
                    kp.append((x,y))

    return kp
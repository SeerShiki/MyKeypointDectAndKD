import MyKeypointDetection
import cv2
import MyKeypointMatch
import PicAdjust

train_img_path = r'D:\PythonProjects\FASTandKD\pictures\magdalen_000951.jpg'
test_img_path  = r'D:\PythonProjects\FASTandKD\pictures\magdalen_000952.jpg'
test_img_path1  = r'D:\PythonProjects\FASTandKD\pictures\magdalen_000951.jpg'
test_img_path2  = r'D:\PythonProjects\FASTandKD\pictures\magdalen_000952.jpg'
Teacher_model_path = r'D:\PythonProjects\FASTandKD\Teacher.pth'
Student_model_path = r'D:\PythonProjects\FASTandKD\Student.pth'
bright = 20
sharpen = 0
shape = (800, 600)
T = 0.05
r = 3
angle_deg = 22.5

rnn = 1
rnn_param_teacher = [32]
rnn_param_student = [16]

transformer = 2
transformer_param_teacher = [32,8,2,0.1]
transformer_param_student = [16,4,2,0.1]

batch_size = 64
epoch = 20
t = 0.5
temprature = 2
alpha = 0.5
device = "cuda"

num_pairs = 512
patch_size = 31

img1 = cv2.imread(test_img_path1, cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, shape)
img1 = PicAdjust.adjust_brightness(img1, 0)
# img1 = PicAdjust.sharpen_image(img1)
IMG1 = img1

img2 = cv2.imread(test_img_path2,cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2,shape)
img2 = PicAdjust.adjust_brightness(img2, 0)
# img2 = PicAdjust.sharpen_image(img2)
IMG2 = img2

# MyKeypointDetection.train_Teacher(IMG1, T, r, angle_deg, transformer, transformer_param_teacher,
#                                   batch_size, epoch, device)
# MyKeypointDetection.train_Student(IMG1, T, r, angle_deg, transformer, transformer_param_teacher,
#                                   rnn, rnn_param_student, batch_size, epoch,temprature, alpha, device)
kp1 = MyKeypointDetection.Predict(IMG1, T, r, angle_deg, rnn, rnn_param_student,
                                     Student_model_path, t, device)
kp2 = MyKeypointDetection.Predict(IMG2, T, r, angle_deg, rnn, rnn_param_student,
                                     Student_model_path, t, device)

# cv2.imshow('image1', IMG1)
# cv2.imshow('image2', IMG2)
# cv2.waitKey(0)
#
rand_pairs = MyKeypointMatch.get_random_pairs(num_pairs, patch_size)
descriptors1 = MyKeypointMatch.generate_brief_descriptors(kp1, img1, rand_pairs, patch_size)
descriptors2 = MyKeypointMatch.generate_brief_descriptors(kp2, img2, rand_pairs, patch_size)
img = MyKeypointMatch.match(img1, img2, shape, kp1, kp2, descriptors1, descriptors2, 50, 2)
cv2.imshow('image', img)
cv2.waitKey(0)
import MyKeypointDetection
import cv2
import MyKeypointMatch

train_img_path = r'D:\PythonProjects\FASTandKD\pictures\9.jpg'
test_img_path  = r'D:\PythonProjects\FASTandKD\pictures\9.jpg'
test_img_path1  = r'D:\PythonProjects\FASTandKD\pictures\9.jpg'
test_img_path2  = r'D:\PythonProjects\FASTandKD\pictures\10.jpg'
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
transformer_param_teacher = [64,4,2,0.1]
transformer_param_student = [32,2,2,0.1]

batch_size = 64
epoch = 20
t = 0.5
temprature = 2
alpha = 0.5
device = "cuda"

num_pairs = 256
patch_size = 15

img1 = cv2.imread(test_img_path1,cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1,shape)
IMG1 = img1
img2 = cv2.imread(test_img_path2,cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2,shape)
IMG2 = img2

MyKeypointDetection.train_Teacher(train_img_path, bright, sharpen, shape, T, r, angle_deg,
                                  rnn, rnn_param_teacher, batch_size, epoch, device)
MyKeypointDetection.train_Student(train_img_path, bright, sharpen, shape, T, r, angle_deg,
                                  rnn, rnn_param_teacher, rnn_param_student, batch_size, epoch,temprature, alpha, device)
_, kp1 = MyKeypointDetection.Predict(test_img_path1, bright, sharpen, shape, T, r, angle_deg,
                                      rnn, rnn_param_student, Student_model_path, t, device)
_, kp2 = MyKeypointDetection.Predict(test_img_path2, bright, sharpen, shape, T, r, angle_deg,
                                      rnn, rnn_param_student, Student_model_path, t, device)

rand_pairs = MyKeypointMatch.get_random_pairs(num_pairs, patch_size)
descriptors1 = MyKeypointMatch.generate_brief_descriptors(kp1, img1, rand_pairs, patch_size, bright)
descriptors2 = MyKeypointMatch.generate_brief_descriptors(kp2, img2, rand_pairs, patch_size, bright)
img = MyKeypointMatch.match(img1, img2, shape, kp1, kp2, descriptors1, descriptors2, 50, 2)
cv2.imshow('image', img)
cv2.waitKey(0)
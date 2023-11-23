
import onnx
import onnxruntime
import numpy as np
import cv2
import time

session = onnxruntime.InferenceSession("fusion7.onnx",
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
img1 = cv2.imread('./test_imgs/vi/17.png',0)
img1 = cv2.resize(img1,(640,480))
img1 = np.float32(img1) / np.float32(255)
#img1 = np.transpose(img1, (2, 0, 1))
img1 = np.expand_dims(img1, 0)
img1 = np.expand_dims(img1, 0)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('./test_imgs/ir/17.png',0)
img2 = cv2.resize(img2,(640,480))
img2 = np.float32(img2) / np.float32(255)
img2 = np.expand_dims(img2, 0)
img2 = np.expand_dims(img2, 0)

ort_input = {session.get_inputs()[0].name: img1.astype(np.float32), session.get_inputs()[1].name: img2.astype(np.float32)}
start = time.time()
ort_output = session.run(None, ort_input)
end = time.time()
print((end - start))
ort_output = np.array(ort_output)
ort_output = ort_output[0][0][0]
img_out = np.float32(ort_output) * np.float32(255)
img_out = np.uint8(img_out)
cv2.imwrite('a.jpg', img_out)

'''
import onnx
import onnxruntime
import numpy as np
import cv2
import time

session = onnxruntime.InferenceSession("fusion5_1.onnx",
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
img1 = cv2.imread('./test_imgs/vi/00858N.png')
img1 = cv2.resize(img1,(400,320))
img1 = np.float32(img1) / np.float32(255)
img1 = np.transpose(img1, (2, 0, 1))
Y = 0.299 * img1[2] + 0.587 * img1[1] + 0.114 * img1[0]
Cr = (img1[2] - Y) * 0.713 + 0.5
Cb = (img1[0] - Y) * 0.564 + 0.5
Y = np.expand_dims(Y, 0)
Y = np.expand_dims(Y, 0)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('./test_imgs/ir/00858N.png',0)
img2 = cv2.resize(img2,(400,320))
img2 = np.float32(img2) / np.float32(255)
img2 = np.expand_dims(img2, 0)
img2 = np.expand_dims(img2, 0)

ort_input = {session.get_inputs()[0].name: Y.astype(np.float32), session.get_inputs()[1].name: img2.astype(np.float32)}
start = time.time()
ort_output = session.run(None, ort_input)
end = time.time()
print((end - start))
ort_output = np.array(ort_output)
ort_output = ort_output[0][0][0]
r = ort_output + (Cr - 0.5)*1.402
g = ort_output - (Cr - 0.5)*0.714 - (Cb - 0.5)*0.344
b = ort_output + (Cb - 0.5)*1.733
img_out = cv2.merge([b,g,r])
img_out = cv2.resize(img_out, (400,320))
img_out = np.float32(img_out) * np.float32(255)
img_out = np.uint8(img_out)
cv2.imwrite('a.jpg', img_out)
'''

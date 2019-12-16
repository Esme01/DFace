from dface.prepare_data.widerface_annotation_gen.wider_loader import WIDER
import cv2
import time

# wider face original images path
path_to_image = 'D:/Junior/人工神经网络/pytorch-face/dataset/WIDER_train/images'

# matlab file path
file_to_label = 'D:/Junior/人工神经网络/pytorch-face/DFace/dface/prepare_data/widerface_annotation_gen/wider_face_train.mat'

#target file path
target_file = 'D:/Junior/人工神经网络/pytorch-face/DFace/dface/prepare_data/widerface_annotation_gen/anno.txt'

wider = WIDER(file_to_label, path_to_image)


line_count = 0
box_count = 0

print('start transforming....')
t = time.time()

with open(target_file, 'w+') as f:
    # press ctrl-C to stop the process
    for data in wider.next():
        line = []
        line.append(str(data.image_name))
        line_count += 1
        for i,box in enumerate(data.bboxes):
            box_count += 1
            for j,bvalue in enumerate(box):
                line.append(str(bvalue))

        line.append('\n')

        line_str = ' '.join(line)
        f.write(line_str)

st = time.time()-t
print('end transforming')

print('spend time:%ld'%st)
print('total line(images):%d'%line_count)
print('total boxes(faces):%d'%box_count)
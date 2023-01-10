import numpy as np
import cv2
import os
import os.path as osp
import imageio

class DataPreprocessing:
    def __init__(self, config=None):
        self.config = config
    
        self.img_id = 0 
        self.q1 = [1, 5, 10, 20, 50]
        self.q2 = [12, 14, 16, 18]

        self.org_path = '/data1/COCO/images/val2017'
        self.dst_path = '/data1/COCO/images/val2017_'
        self.blur_path = '/data1/COCO/images/val2017_b'
        self.dark_path = '/data1/COCO/images/val2017_d'


    def make_patch(self, imgs):
        tmpimg = imgs[image_id-start_id].copy()
        tmpimg = tmpimg.astype('uint8')
        cv2.imwrite(osp.join(self.config.patch_dir, cropped_data[0]['imgpath'][:-4] + '_' + str(img_id) + '.jpg'), tmpimg)
        self.img_id += 1

    def make_blur_bbox(self, cropped_data):
        file = cv2.imread(os.path.join(self.config.img_path, cropped_data[0]['imgpath']))
        for data in cropped_data:
            num_keypoints = data['num_keypoints']
            if num_keypoints > 0:
                xb, yb, wb, hb = data['bbox']
                xb = math.trunc(xb); yb = math.trunc(yb); wb = math.trunc(wb); hb = math.trunc(hb)
                mosaic_loc = file[yb:yb+hb, xb:xb+wb]
                mosaic_loc = cv2.GaussianBlur(mosaic_loc, (5,5), 1)
                # file_mosaic = file
                file[yb:yb+hb, xb:xb+wb] = mosaic_loc
            cv2.imwrite(osp.join(self.config.blur_dir, cropped_data[0]['imgpath'][-16:-4] + '.jpg'), file)

    def make_blur(self):
        files = os.listdir(self.org_path)

        for file in files:
            # f = Image.open(os.path.join(self.org_path, file))
            # blur = f.filter(ImageFilter.BLUR)
            
            # blur.save(osp.join(self.blur_path, file))
            f = cv2.imread(os.path.join(self.org_path, file), cv2.IMREAD_COLOR)
            cv2.imwrite(osp.join(self.blur_path, file), sub)

    def change_filename(self):
        files = os.listdir(self.org_path)
        files.sort()
        i = 0

        for file in files:
            src = os.path.join(self.org_path, file)
            dst = str(i) + '.jpg'
            dst = os.path.join(self.org_path, dst)
            os.rename(src, dst)
            i += 1

    def lower_quality(self):
        # for i in self.q1:
        for i in self.q2:
            files = os.listdir(self.org_path)
            os.mkdir(self.dst_path+str(i))
            for file in files:
                img = imageio.imread(osp.join(self.org_path, file))
                imageio.imwrite(osp.join(self.org_path+'_'+str(i), file), img, quality = i)

    def make_dark(self):
        for i in q1:
            files = os.listdir(self.org_path+str(i))
            for file in files:
                f = cv2.imread(os.path.join(self.org_path+str(i), file), cv2.IMREAD_COLOR)
                array = np.ones(f.shape, dtype="uint8") * 100
                sub = cv2.subtract(f, array)
                cv2.imwrite(osp.join(self.dark_path+str(i), file), sub)

    

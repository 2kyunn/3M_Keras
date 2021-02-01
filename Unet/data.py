from skimage.io import imread
from skimage.color import rgba2rgb
import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize
from os import listdir
from os.path import isfile, join

################################ DATA class ################################################################
######## function : get_imgpath  --> 원본 data imread(경로) 로 return , if rgba --> rgb #####################
##################  get_maskpath  --> mask data imread(경로) 로 return , if rgba --> rgb #####################
##################  random_crop --> org_image와 mask_image를 crop하여 target_size로 변환 ###############
##################  rgb2label   --> (H, W, 3)-> (H, W, # of class) : 원핫인코딩 -> 네트워크 출력 y 형태로 변환##################
##################  preprocess_input --> rgb2label + random_crop             ###############################
##################  image_generator  --> batch


class DATA:
    def __init__(
        self,
        data_path,
        target_size=(256, 512),
        batch_size=16,
        rescale=1 / 255,
        random_cropping=False,
    ):
        def get_imgpath(data_path):
            image = imread(data_path)
            if image.shape[2] == 4:
                image = rgba2rgb(image)

            return image

        def get_maskpath(org_path):
            mask_id = org_path.split("\\")[-1].split("leftImg8bit")[0]
            # file name before 'leftImg8bit'
            mask_id += "gtFine_color.png"
            mask_path = org_path.split("images")[0]
            mask_path = mask_path + "masks/mask/" + mask_id
            image = imread(mask_path)
            return image

        def random_crop(org_image, mask_image, target_size):
            # Note: image_data_format is 'channel_last'
            assert org_image.shape[2] == 3
            height, width = org_image.shape[0], org_image.shape[1]
            dy, dx = target_size
            x = np.random.randint(0, width - dx + 1)
            y = np.random.randint(0, height - dy + 1)

            return (
                org_image[y : (y + dy), x : (x + dx), :],
                mask_image[y : (y + dy), x : (x + dx), :],
            )

        def rgb2label(img, color_codes=None, one_hot_encode=False):
            if color_codes is None:
                color_codes = {
                    val: i
                    for i, val in enumerate(set(tuple(v) for m2d in img for v in m2d))
                }
            n_labels = len(color_codes)
            result = np.ndarray(shape=img.shape[:2], dtype=int)
            result[:, :] = -1
            for rgb, idx in color_codes.items():
                result[(img == rgb).all(2)] = idx

            if one_hot_encode:
                one_hot_labels = np.zeros((img.shape[0], img.shape[1], n_labels))
                # one-hot encoding
                for c in range(n_labels):
                    one_hot_labels[:, :, c] = (result == c).astype(int)
                result = one_hot_labels

            return result, color_codes

        #### random cropping + rgb2label ####
        def preprocess_input(org_image, mask_image, target_size, rescale):
            org_image = img_as_ubyte(
                resize(org_image, target_size, order=0)
            )  ### order = 0 : image value 가 class probability (0,1) 이므로, 다른 이상한 상수가 resize 할때 안 나오도록 함
            mask_image = img_as_ubyte(resize(mask_image, target_size, order=0))

            if random_cropping:
                org_image, mask_image = random_crop(org_image, mask_image, target_size)

            color_codes = {(255, 0, 0): 0, (255, 0, 255): 1}
            mask_image, _ = rgb2label(mask_image, color_codes, one_hot_encode=True)

            org_image = org_image.astype("float32")
            mask_image = mask_image.astyp("float32")

            if rescale:
                org_image *= rescale
                #### 0 ~ 255 -> 0 ~ 1 rescale

            return org_image, mask_image

        def image_generator(files, batch_size):
            while True:
                batch_paths = np.random.choice(
                    files, size=batch_size, replace=False
                )  ## : 비 복원 샘플링 : 16(batch_size) 개 중에서는 겹치는 것 x
                batch_input = []
                batch_output = []

                for org_path in batch_paths:
                    org_data = get_imgpath(org_path)
                    mask_data = get_maskpath(org_path)

                    org_data, mask_data = preprocess_input(
                        org_data, mask_data, target_size, rescale=rescale
                    )
                    batch_input += [org_data]
                    batch_output += [mask_data]

                batch_x = np.array(batch_input)
                batch_y = np.array(batch_output)

                yield (batch_x, batch_y)

        print(data_path)
        files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        print(files)
        files = [data_path + x for x in files]
        print(files)
        self.image_generator = image_generator(files, batch_size=batch_size)


# a = DATA("C:\\Users\\Jungyun\\Desktop\\3mkeras\\datasets\\a\\")

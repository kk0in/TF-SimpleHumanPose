class Deburger:
    def __init__(self, config=None):
        self.config = config

    def d_one(self, cropped_data, batch_id, imgs, crop_infos, heatmap):
        print("cropped_data: ", cropped_data)
        print("length of cropped data: ", len(cropped_data))
        print("batch_id: ", batch_id)
        print("shape of imgs: ", imgs.shape)
        print("imgs: ", imgs)
        print("shape of crop_infos: ", crop_infos.shape)
        print("crop_infos: ", crop_infos)
        print("shape of heatmap: ", heatmap.shape)
        print("heatmap: ", heatmap)
        exit()  

    def d_two(self, kps_result):
        print("len(kps_result):", len(kps_result))
        print("shape(kps_result):", kps_result.shape)
        print("shape(kps_result[:,:,2]):", kps_result[:,:,2].shape)
        print("kps_result[:,:,2]", kps_result[:,:,2])

    def d_three(self, hm_j, image_id, y, x, py, px, diff):
        print("hm_j shape:", hm_j.shape)
        print("hm_j:", hm_j)
        print("hm_j.max:", hm_j.max())
        print("image_id:", image_id)
        print("y, x: ", y, x)
        print("py, px: ", py, px)
        print("diff: ", diff)
        exit()

    def d_four(self, kps_result, image_id):
        print("kps_result[image_id,:,2]:", kps_result[image_id,:,2])
        print("np.any(kps_result[image_id,:,2])", int(np.any(kps_result[image_id,:,2])))

        




from data.demo_dataset import DemoDataset

class DemoAppearanceDataset(DemoDataset):
    def __init__(self, data_root, opt, load_from_dataset=False):
        super(DemoAppearanceDataset, self).__init__(data_root, opt, load_from_dataset)    

    def load_item(self, garment_img_path, reference_img_path, label_path=None):
        if self.load_from_dataset:
            reference_img_path = self.transfrom_2_real_path(reference_img_path)
            garment_img_path = self.transfrom_2_real_path(garment_img_path)
            label_path = self.transfrom_2_real_path(label_path)
        else:
            reference_img_path = self.transfrom_2_demo_path(reference_img_path)
            garment_img_path = self.transfrom_2_demo_path(garment_img_path)
            label_path = self.transfrom_2_demo_path(label_path)

        label_path = self.img_to_label(label_path)   
        reference_img = self.get_image_tensor(reference_img_path)[None,:]
        garment_img = self.get_image_tensor(garment_img_path)[None,:]

        label, face_center = self.get_label_tensor(label_path)

        garment_label_path = self.img_to_label(garment_img_path)
        _, garment_face_center = self.get_label_tensor(garment_label_path)
        return {'reference_image':reference_img, 
                'garment_image':garment_img, 
                'target_skeleton':label[None,:], 
                'face_center':face_center[None,:],
                'garment_face_center':garment_face_center[None,:],
                }
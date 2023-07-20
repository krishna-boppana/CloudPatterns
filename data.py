from torchvision import transforms
from torch.utils.data import Dataset
from torch import tensor
from PIL import Image
import pandas as pd
from torchvision.transforms import functional
class CloudsDataset(Dataset):
    def __init__(self, labels_csv:str , image_directory:str, augmentation = None):
        self.just_labels = False
        self.augmentation = augmentation
        self.image_directory = image_directory
        self.label_df = self.construct_data_frame(labels_csv)
        self.image_names = list(self.label_df.index.unique())

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx:int):
        img_name = self.image_names[idx]
        img_labels = self.encode_labels(img_name)

        if self.just_labels:
            return img_labels
        
        else:
            img = Image.open(self.image_directory + img_name)
            if self.augmentation:
                img = transforms.ToTensor()(img)
                img = self.augmentation(img)
            else:
                img = transforms.ToTensor()(img)
                
            return img, img_labels
        
    def construct_data_frame(self,path_str:str):

        labels = pd.read_csv(path_str)
        # adding columns to data frame to split by image name and class_label (example 0011165.jpg_Fish - > [0011165.jpg, Fish])
        labels[["image_name","class_label"]] = labels.Image_Label.str.split('_',expand=True)

        #setting index to image_id column because df.loc() is a constant time operation
        labels = labels.set_index('image_name')

        labels = labels.fillna("",axis=1)
        
        return labels
    
    def encode_labels(self,name:str):
        img_labels = self.label_df.loc[name].set_index("class_label").drop(columns = "Image_Label")
        
        encoded_labels = [0,0,0,0]

        #Determining if label is present in image
        #4 types of shallow cloud formations (fish,flower,gravel,sugar)
        encoded_labels[0] = (img_labels.loc["Fish","EncodedPixels"] != "" )
        encoded_labels[1] = (img_labels.loc["Flower","EncodedPixels"] != "")
        encoded_labels[2] = (img_labels.loc["Gravel","EncodedPixels"] != "") 
        encoded_labels[3] = (img_labels.loc["Sugar","EncodedPixels"] != "")
        
        label = tensor(encoded_labels)
        return label.float()

class resize_maintain_aspect_ratio:
    def __init__(self, width = 224, height = 224):
        self.width = width
        self.height = height

    def __call__(self, image):
        '''
        img_width = 2100
        img_height = 1400
        '''
        num_padding = 350 #(width - height / 2)

        #padding height dimension
        image = functional.pad(image,(0,num_padding,0,num_padding))
        image = functional.resize(image,size = (self.width,self.height))
        return image
    
    #training augmentation
augmentation = transforms.Compose([
    resize_maintain_aspect_ratio(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((0,30)),
    transforms.ColorJitter(brightness=(.8,1.2)),
    transforms.Grayscale(num_output_channels=3),
    #ImageNet Stats
    transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ]
)

#basic augmentation
augmentation_basic = transforms.Compose([
    resize_maintain_aspect_ratio(),
    transforms.Grayscale(num_output_channels=3),
    # #ImageNet Stats
    transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ]
)
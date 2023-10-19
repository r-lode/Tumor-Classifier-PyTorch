import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class Custom_Image_Dataset(Dataset):
    '''
    Custom PyTorch image data handler for tumor classification
    '''

    def __init__(self, image_directory, transform = None):
        '''
        Initialize image dataset

        Args:
            image_directory (str): Path to directory containing images.
            transform(callable, function): Default = None. Function imput to apply to
            images
        
        '''

        self.image_dir = image_directory
        self.classes = os.listdir(image_directory) #Get subdirectory class names
        self.image_paths = []
        self.transform = transform

        #Class mapping:
        self.class_mapping = {
            "no_tumor": 0,
            "glioma_tumor": 1,
            "meningioma_tumor": 2,
            "pituitary_tumor": 3

        }

        self.reverse_class_mapping = {
            v : k for k, v in self.class_mapping.items()
        }

        #store numeric labels
        self.numeric_class_labels = [] 

        for tumor_class in self.classes:
            class_directory = os.path.join(image_directory, tumor_class)
            image_files = os.listdir(class_directory)
            self.image_paths.extend([os.path.join(tumor_class, image) for image in image_files])
            self.numeric_class_labels.extend([self.class_mapping[tumor_class]] * len(image_files))

    
    def __len__(self):
        '''
        Return the number of images in the data
        '''
        return len(self.image_paths)
    
    def __getitem__(self, index):
        '''
        Returns image and its corresponding tumor class numeric label

        Args: 
            index(int): Sample  index
        
            Returns:
                tuple of image and numeric label
        '''
        #get image path 
        path_to_image = os.path.join(self.image_dir, self.image_paths[index])
        image = Image.open(path_to_image)

        #transform image if a transform is being applied to all images
        if self.transform:
            image = self.transform(image)
    
        #get numeric label
        label = self.numeric_class_labels[index]

        return image, label
    
    def get_class_string_label(self, num_label):
        '''
        Convert tumor class numeric label to its mapped class name.

        Args: 
            num label(int): numeric label to map to class string

        Returns: 
            str: class name in string format
        '''
        return self.reverse_class_mapping[num_label]

    
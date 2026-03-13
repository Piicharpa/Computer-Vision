import torchvision.transforms as transforms

def get_transform(train=True):

    if train:

        transform = transforms.Compose([

            transforms.Resize((256,256)),

            transforms.RandomResizedCrop(224),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(15),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )

        ])

    else:

        transform = transforms.Compose([

            transforms.Resize((224,224)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )

        ])

    return transform
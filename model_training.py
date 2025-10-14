from ultralytics import YOLO
import torch

def main():
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    #shortcut names for different models available in repo
    yolov5su = "yolov5su.pt"
    yolov8s = "yolov8s.pt"
    yolov8m = "yolov8m.pt"
    yolov11n = "yolo11n.pt"
    yolov11s = "yolo11s.pt"
    yolov11m = "yolo11m.pt"
    yolov11l = "yolo11l.pt"
    custom_yolo = "my_training_runs/yolo11l_ua_detrac_2002/weights/last.pt" #Change this for different custom trained models
    rtdetr_l = "rtdetr-l.pt" #RTDETR large model

    model = YOLO("/home/yanjiaqi/own_ultralytics/ultralytics/my_training_runs/yolo11m_bdd100k_100/weights/best.pt")  # load a pretrained model (recommended for transfer learning)

    #=================================================For training on ua_detrac dataset==========================================================#
    # results = model.train(
    #     data = "custom.yaml",    #"ultralytics/cfg/datasets/the-yaml-used",
    #     epochs = 100,
    #     imgsz = 800,
    #     batch = 16,
    #     lr0 = 0.001,
    #     weight_decay = 0.0005,
    #     amp = True,
    #     augment= True,
    #     hsv_h= 0.015,  # Randomly adjust hue
    #     hsv_s= 0.7,    # Randomly adjust saturation - helps with lighting changes
    #     hsv_v= 0.4,    # Randomly adjust value
    #     translate= 0.1,  # Randomly translate images by up to 10%
    #     scale= 0.5,    # Randomly scale images by up to 50% - crucial for size variance
    #     shear= 0.0,    # Shear is less critical for vehicles, can keep low
    #     flipud= 0.0,   # Flip up-down (usually not logical for traffic scenes)
    #     fliplr= 0.5,  # Flip left-right - very logical and effective
    #     mosaic= 1.0,   # Use mosaic augmentation (combines 4 images) - keep enabled
    #     mixup= 0.0,    # Start with mixup off, can try 0.1 later if needed
    #     cos_lr = True,  # Use cosine learning rate schedule for smoother training
    #     patience = 25,  # Early stopping patience
    #     optimizer = 'AdamW',
    #     momentum = 0.9,
    #     project = "my_training_runs",
    #     name = "yolo11m_bdd100k_100",
    #     freeze = 0, 
    #     save_period = 20, #Saves a checkpoint every 20 epochs 
    #     save = True,
    #     exist_ok = False #overrides if folder with same name already exists. REMEMBER TO CHANGE TO FALSE WHEN DOING ACTUAL NEW TRAINING

    # )

    #=================================================For fine-tuning using bdd100k subset dataset==========================================================#
    results = model.train(
        data="custom.yaml",
        warmup_epochs=3.0,      # Shorter warmup
        epochs=50,              # Reduced from 100
        imgsz=800,
        batch=16,
        lr0=0.0001,            # 10x lower than initial training
        weight_decay=0.0005,
        amp=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        cos_lr=True,
        patience=15,           # Reduced patience
        optimizer='AdamW',
        momentum=0.9,
        project="my_training_runs",
        name="yolo11m_bdd100k_finetune",  # Different name
        freeze=10,             # Freeze first 10 layers
        save_period=10,
        save=True,
        exist_ok=False
    )

    #==============================================Planned for rtdetr training=======================================================#
    # results = model.train(
    #     data = "ua_detrac_10k.yaml",    #"ultralytics/cfg/datasets/ua_detrac.yaml",
    #     epochs = 100,
    #     imgsz = 800,
    #     batch = 8,
    #     lr0 = 0.0001,
    #     weight_decay = 0.0005,
    #     amp = True,
    #     augment= True,
    #     hsv_h= 0.015,  # Randomly adjust hue
    #     hsv_s= 0.7,    # Randomly adjust saturation - helps with lighting changes
    #     hsv_v= 0.4,    # Randomly adjust value
    #     translate= 0.1,  # Randomly translate images by up to 10%
    #     scale= 0.5,    # Randomly scale images by up to 50% - crucial for size variance
    #     shear= 0.0,    # Shear is less critical for vehicles, can keep low
    #     flipud= 0.0,   # Flip up-down (usually not logical for traffic scenes)
    #     fliplr= 0.5,  # Flip left-right - very logical and effective
    #     mosaic= 1.0,   # Use mosaic augmentation (combines 4 images) - keep enabled
    #     mixup= 0.0,    # Start with mixup off, can try 0.1 later if needed
    #     cos_lr = True,  # Use cosine learning rate schedule for smoother training
    #     patience = 25,  # Early stopping patience
    #     optimizer = 'AdamW',
    #     momentum = 0.9,
    #     project = "my_training_runs",
    #     name = "rtdetr_ua_detrac_10k_100",
    #     warmup_epochs= 5.0, # (float) warmup epochs (fractions allowed)
    #     warmup_momentum= 0.9, # (float) initial momentum during warmup
    #     warmup_bias_lr= 0.1, # (float) bias learning rate during warmup
    #     freeze = 0, 
    #     save_period = 20, #Saves a checkpoint every 20 epochs 
    #     save = True,
    #     exist_ok = False #overrides if folder with same name already exists. REMEMBER TO CHANGE TO FALSE WHEN DOING ACTUAL NEW TRAINING
    # )

    # #===============================================For resuming training=======================================================#
    # results = model.train(resume = True)


if __name__ == "__main__":
    main()
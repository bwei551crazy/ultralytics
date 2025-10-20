from ultralytics import YOLO
import torch

def main():
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    #shortcut names for different models available in repo
    yolov5su = "yolov5su.pt"
    yolov8s = "yolov8s.pt"
    yolov8m = "yolov8m.pt"
    yolo11n = "yolo11n.pt"
    yolo11s = "yolo11s.pt"
    yolo11m = "yolo11m.pt"
    yolo11l = "yolo11l.pt"
    custom_yolo = "my_training_runs/yolo11l_ua_detrac_2002/weights/last.pt" #Change this for different custom trained models
    rtdetr_l = "rtdetr-l.pt" #RTDETR large model

    model = YOLO("/home/yanjiaqi/own_ultralytics/ultralytics/my_training_runs/rtdetr_bdd100k_150_orig_lbl/weights/best.pt")  # load a pretrained model (recommended for transfer learning)

    #=================================================Fine-tuning dataset==========================================================#
    results = model.train(
        data="fine_tune.yaml",
        epochs = 80,
        patience = 20,
        warmup_epochs = 5.0,

        lr0 = 0.00003,
        lrf = 0.001,
        cls = 1.0,
        box = 5.0,

        mosaic = 0.4,
        copy_paste = 0.1,

        hsv_h = 0.01,
        hsv_s = 0.3,
        hsv_v = 0.2,

        degrees = 0.0,
        translate = 0.05,
        scale = 0.2,
        shear = 0.0,
        perspective = 0.0,

        fliplr = 0.5,
        flipud = 0.0,

        freeze = 10,
        optimizer = 'AdamW',

        project = "my_training_runs",
        name = "rtdetr_bdd100k_150_orig_lbl_finetune",
        save_period = 20,
        save = True,
        exist_ok = False


    )

    #==============================================Planned for rtdetr training=======================================================#
    # results = model.train(
    #     data = "custom.yaml",    #"ultralytics/cfg/datasets/ua_detrac.yaml",
    #     epochs = 150,
    #     imgsz = 800,
    #     batch = 8,
    #     lr0 = 0.0001,
    #     lrf = 0.01,
    #     weight_decay = 0.0001,
    #     amp = True,
    #     augment= True,
    #     hsv_h= 0.05,  # Randomly adjust hue
    #     hsv_s= 0.5,    # Randomly adjust saturation - helps with lighting changes
    #     hsv_v= 0.4,    # Randomly adjust value
    #     translate= 0.05,  # Randomly translate images by up to 10%
    #     scale= 0.5,    # Randomly scale images by up to 50% - crucial for size variance
    #     shear= 0.0,    # Shear is less critical for vehicles, can keep low
    #     flipud= 0.0,   # Flip up-down (usually not logical for traffic scenes)
    #     fliplr= 0.5,  # Flip left-right - very logical and effective
    #     mosaic= 0.5,   # Use mosaic augmentation (combines 4 images) - keep enabled
    #     mixup= 0.0,    # Start with mixup off, can try 0.1 later if needed
    #     copy_paste = 0.0,
    #     cos_lr = True,  # Use cosine learning rate schedule for smoother training
    #     patience = 25,  # Early stopping patience
    #     optimizer = 'AdamW',
    #     momentum = 0.9,
    #     project = "my_training_runs",
    #     name = "rtdetr_bdd100k_150_orig_lbl",
    #     warmup_epochs= 5.0, # (float) warmup epochs (fractions allowed)
    #     warmup_momentum= 0.9, # (float) initial momentum during warmup
    #     warmup_bias_lr= 0.1, # (float) bias learning rate during warmup
    #     freeze = 12, 
    #     save_period = 20, #Saves a checkpoint every 20 epochs 
    #     save = True,
    #     exist_ok = False #overrides if folder with same name already exists. REMEMBER TO CHANGE TO FALSE WHEN DOING ACTUAL NEW TRAINING
    # )

    model = YOLO(yolo11l)  # load a pretrained model (recommended for transfer learning)


        #=================================================Planned for YOLO training ==========================================================#
    results = model.train(
        data = "custom.yaml",    #"ultralytics/cfg/datasets/the-yaml-used",
        epochs = 150,
        warmup_epochs = 5.0,
        imgsz = 800,
        batch = 8,
        lr0 = 0.0001,
        lrf = 0.01,
        weight_decay = 0.0001,
        amp = True,
        augment= True,
        hsv_h= 0.015,  # Randomly adjust hue
        hsv_s= 0.5,    # Randomly adjust saturation - helps with lighting changes
        hsv_v= 0.4,    # HSV value (brightness) augmentation fraction
        translate= 0.05,  # Randomly translate images by up to 10%
        scale= 0.3,    # Randomly scale images by up to 50% - crucial for size variance
        shear= 0.0,    # Shear is less critical for vehicles, can keep low
        flipud= 0.0,   # Flip up-down (usually not logical for traffic scenes)
        fliplr= 0.5,  # Flip left-right - very logical and effective
        degrees = 0.0,
        copy_paste = 0.1,
        mosaic= 0.5,   # Use mosaic augmentation (combines 4 images) - keep enabled
        mixup= 0.0,    # Start with mixup off, can try 0.1 later if needed
        cos_lr = True,  # Use cosine learning rate schedule for smoother training
        patience = 25,  # Early stopping patience
        optimizer = 'AdamW',
        momentum = 0.9,
        project = "my_training_runs",
        name = "yolo11l_bdd100k_150_orig_lbl",
        freeze = 10, 
        save_period = 20, #Saves a checkpoint every 20 epochs 
        box = 7.5,# (float) box loss gain
        cls = 0.5, # (float) classification loss gain
        dfl = 1.5, # (float) distribution focal loss gain
        save = True,
        exist_ok = False #overrides if folder with same name already exists. REMEMBER TO CHANGE TO FALSE WHEN DOING ACTUAL NEW TRAINING

    )
    

    #===============================================For resuming training=======================================================#
    # results = model.train(resume = True)


if __name__ == "__main__":
    main()
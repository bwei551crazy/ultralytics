from ultralytics import YOLO

def main():
    
    #shortcut names for different models available in repo
    yolov5su = "yolov5su.pt"
    yolov8s = "yolov8s.pt"
    yolov11 = "yolov11s.pt"
    custom_yolo = "my_training_runs/yolov8s_1.pt/weights/best.pt" #Change this for different custom trained models

    #loading a pretrained model
    model = YOLO(yolov5su)

    results = model.train(
        data = "VisDrone.yaml",
        epochs = 200,
        imgsz = 640,
        batch = 16,
        lr0 = 0.001,
        lrf = 0.01,
        # degrees = 5.0,
        # translate = 0.2,
        # scale = 0.3,
        project = "my_training_runs",
        name = "yolov5su_visdrone_200",
        save = True,
        exist_ok = False #overrides if folder with same name already exists

    )

if __name__ == "__main__":
    main()
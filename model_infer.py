from ultralytics import YOLO

def main():

    model = YOLO("my_training_runs/yolo11l_coco128_300/weights/best.pt") #Change this to your custom trained model path
    results = model.predict(
        source = "data/images/Kyoto walk.mp4", 
        conf = 0.89, #add a parameter called conf to add confidence cutoffs
        save = True,
        project = "my_training_runs/yolo11l_coco128_300_infer",
        name = "Kyotowalk_infer",
        show = True,                                #display vid during processing
        verbose = True,                              #show progress
        exist_ok = False
    ) 
    

    # results[0].show() #uses matplotlib

if __name__ == "__main__":
    main()
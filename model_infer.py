from ultralytics import YOLO

def main():

    model = YOLO("my_training_runs/yolo11l_coco128_300/weights/best.pt") #Change this to your custom trained model path
    results = model.predict(
        source = "datasets/ua-detrac (original)/images/test", 
        conf = 0.89, #add a parameter called conf to add confidence cutoffs
        save = True,
        project = "my_training_runs/yolo11l_coco128_300_infer",
        name = "ua_detrac_ifer",
        #show = True,                                #display vid during processing
        verbose = True,                              #show progress
        exist_ok = False
    ) 
    

    # results[0].show() #uses matplotlib

if __name__ == "__main__":
    main()
from ultralytics import YOLO

def main():

    model = YOLO("my_training_runs/yolov5su_custom_hyp_3003/weights/best.pt") #Change this to your custom trained model path
    results = model.predict(
        source = "data/images/20250916_190547.mp4", 
        conf = 0.6, #add a parameter called conf to add confidence cutoffs
        save = True,
        project = "my_training_runs/yolov5su_custom_hyp_3003",
        name = "20250916_190547_infer.mp4",
        show = True,                                #display vid during processing
        verbose = True                              #show progress
        ) 
    

    # results[0].show() #uses matplotlib

if __name__ == "__main__":
    main()
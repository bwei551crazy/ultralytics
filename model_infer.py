from ultralytics import YOLO

def main():

    #Evaluating model inference performance
    model = YOLO("my_training_runs/rtdetr_ua_detrac_10k_1002/weights/best.pt")          #Change this to your custom trained model path
    results = model.predict(
        source = "data/images/20250916_190547.mp4",                                     #Chuck whatever file you want for model inference here. Can be images and video
        conf = 0.20,                                                                    #confidence cutoff point
        save = True,
        project = "my_test_runs/rtdetrl_infer",                                         #Main folder name path
        name = "rtdetrl_ua-detrac_infer",                                               #Sub folder within main folder
        show = False,                                                                   #Display vid during inference. Default is False
        verbose = False,                                                                #show progress. Default is True
        exist_ok = True                                                                 #Overrides if the same inferred file already exists in directory
    ) 
    
    ##Evaluating validation metrics 
    # metrics = model.val()
    # metrics.box.map                                   # map50-95
    # metrics.box.map50                                 # map50
    # metrics.box.map75                                 # map75
    # metrics.box.maps                                  # a list contains map50-95 of each category

    # results[0].show()                                 #uses matplotlib

if __name__ == "__main__":
    main()
from ultralytics import YOLO

def main():

    model = YOLO("/home/yanjiaqi/own_ultralytics/ultralytics/my_training_runs/rtdetrl_bdd100k_150_orig_lbl2_finetune/weights/best.pt")          #Change this to your custom trained model path

    # #Evaluating model inference performance    
    results = model.predict(
        source = "/home/yanjiaqi/own_ultralytics/ultralytics/data/images/Tuen Mun Highway East Bound屯門公路東行 .mp4",                                     #Chuck whatever file you want for model inference here. Can be images and video
        conf = 0.75,                                                                    #confidence cutoff point
        save = True,
        project = "my_test_runs/rtdetrl_infer",                                         #Main folder name path
        name = "rtdetrl_bdd100k_150_orig_lbl_finetune",                                               #Sub folder within main folder
        show = True,                                                                   #Display vid during inference. Default is False
        verbose = False,                                                                #show progress. Default is True
        exist_ok = True                                                                 #Overrides if the same inferred file already exists in directory
    ) 
    
    #Evaluating validation metrics 
    # metrics = model.val()
    # metrics.box.map                                   # map50-95
    # metrics.box.map50                                 # map50
    # metrics.box.map75                                 # map75
    # metrics.box.maps                                  # a list contains map50-95 of each category

    # results[0].show()                                 #uses matplotlib

if __name__ == "__main__":
    main()
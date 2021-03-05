# ScenceSegNet

Structure /<br>
 ../input/data/data/*pkl
 ../input/data/train
 ../input/data/validation
 ../input/data/test
 
 ../input/model
 
 ../predict
 ../src
 
 Step1: 
  put all data in  ../input/data/data
  put prediction.py and config.py in ../predict
  put 
    create_train_val_test.py
    dataset.py
    evaluation.py
    models.py
    run.sh
    train.py
  in ../src
 
 Step2:
  run create_train_val_test.py
  the data will be split in 
    ../input/data/train/*pkl
    ../input/data/validation/*pkl
    ../input/data/test/*pkl
    
  Step3:
   run sh run.sh
   the training history and model wil be in
    ../input/model
    
  Step4:
    run predict.py
    

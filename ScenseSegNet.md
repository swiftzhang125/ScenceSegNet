# ScenceSegNet

## Structure <br>
input <br>
 > data <br>
 >> data <br>
 >> train <br>
 >> validation <br>
 >> test <br>

input<br>
 > model<br>

predict<br>

src<br>


## Steps <br>
  ### Step1:<br> 
   put all data in  ../input/data/data <br>
   put prediction.py and config.py in ../predict<br>
   put<br> 
    * create_train_val_test.py<br>
    * dataset.py<br>
    * evaluation.py<br>
    * models.py<br>
    * run.sh<br>
    * train.py<br>
   in ../src<br>
 <br> 
 ### Step2:<br>
  run create_train_val_test.py<br>
  the data will be split in <br>
    * ../input/data/train/*pkl<br>
    * ../input/data/validation/*pkl<br>
    * ../input/data/test/*pkl<br>
 <br>
 ### Step3:<br>
   run sh run.sh<br>
   the training history and model wil be in<br>
    ../input/model<br>
 <br>   
 ### Step4:<br>
    run predict.py
    

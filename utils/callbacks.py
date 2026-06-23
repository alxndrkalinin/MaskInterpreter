from curses import termattrs
import tensorflow as tf
import tensorflow.keras as keras
import global_vars as gv

class SaveModelCallback(keras.callbacks.Callback):
    
    def __init__(self, freq, model, path=gv.model_path,monitor="val_loss",term=None,term_value=None,save_all=False):
        super().__init__()
        self.freq = freq
        self.model = model
        self.path = path
        self.epoch = 0
        self.min_loss = 10000000
        self.monitor = monitor
        self.term  = term
        self.term_value = term_value
        self.save_all = save_all
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch+=1
        save = False
        print("current {} is :{}, min {} is:{}".format(self.monitor,logs[self.monitor],self.monitor,self.min_loss))
        if (self.epoch % self.freq == 0 and logs[self.monitor]<self.min_loss):
            if ((self.term is not None and self.term_value is not None) or self.save_all):
                if (logs[self.term] > self.term_value) or self.save_all:
                    save = True
            else:
                save = True
            if save:
                print("saving model {}...".format(self.path))
                self.min_loss =  logs[self.monitor]
                try:
                    self.model.save(self.path,save_format="tf")
                except Exception as e:
                    self.model.save_weights(self.path)
                print("model saved.")


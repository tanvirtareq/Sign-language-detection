import threading

from test import Test
import test2

class CameraThread(threading.Thread):
    def __init__(self, obj):
        threading.Thread.__init__(self)
        self.obj=obj
        self.running = False
 
        # helper function to execute the threads
    def stop(self):
        print('stop')
        self.running = False

    def run(self):
        self.running = True
        print(self.obj.image_label.text())
        test2.start(self)
        # t=Test(self.obj)
        # t.start()

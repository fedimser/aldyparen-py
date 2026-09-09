print("start")
import time
t0 = time.time()

from aldyparen.gui.app import AldyparenApp
print("Import done", time.time()-t0)

if __name__ == '__main__':
    AldyparenApp().run()


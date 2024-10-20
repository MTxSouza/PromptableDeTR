"""
Module that allows run the UI data label by simply calling the `label_ui` 
package on terminal as shown below:
```
$ python label_ui
```
"""
import os

if __name__=="__main__":

    # Get `app.py` path.
    app_path = os.path.join(os.path.abspath(path=os.path.dirname(p=__file__)), "app.py")
    assert os.path.exists(path=app_path), "Could not localize the `app.py` module."

    # Run app.
    os.system(command="python %s" % app_path)

# nmr-classifier

*Jisun Kang*

nmr-classifier is a Python (2 and 3) library to implement the nuclear norm regression for classification.
This package uses facial structural information to recognize facial images.

## Install
You can use pip to install nmr-classifier.
```{Python}
pip install nmr_classifier
import nmr_classifier
```

## Resource
- https://pypi.org/project/nmr-classifier/0.1/

## Example
```{Python}
from sklearn.datasets import fetch_olivetti_faces
olivetti = fetch_olivetti_faces()
num=91; n = 400; occlusion_percent = 0.2; 
black_size = round(test_img.shape[1]*occlusion_percent)
loc = random.randint(0,64-black_size)
test_img[loc:loc+black_size,loc:loc+black_size] = np.zeros((black_size,black_size))
target = list(olivetti.target)

from nmr_classifier.fast_admm_nmr_classifier import nmr_classifier
clf = nmr_classifier()
clf.fit(train_img, test_img)
clf.classifier(train_img, test_img, num, target)
```

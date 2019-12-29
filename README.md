# nmr-classifier

*Jisun Kang*

nmr-classifier is a Python (2 and 3) library to implement the nuclear norm regression for classification.
This package uses facial structural information to recognize facial images.

## Install
You can use pip to install nmr-classifier.
```{Python}
# install
pip install nmr_classifier

# update
pip install --update nmr_classifier

# check
import nmr_classifier
```

## Resource
- https://pypi.org/project/nmr-classifier/0.2.3/

## Example
```{Python}
# import olivetti dataset
from sklearn.datasets import fetch_olivetti_faces
olivetti = fetch_olivetti_faces()

# create occlusion
num=91; n = 400; occlusion_percent = 0.2; 
black_size = round(test_img.shape[1]*occlusion_percent)
loc = random.randint(0,64-black_size)
test_img[loc:loc+black_size,loc:loc+black_size] = np.zeros((black_size,black_size))

# create target vector
target = list(olivetti.target)

# import classifier
from nmr_classifier.fast_admm_nmr_classifier import nmr_classifier

# define classifier
clf = nmr_classifier()

# fitting dataset
clf.fit(train_img, test_img)

# classification
clf.classifier(train_img, test_img, num, target)
```

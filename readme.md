# How to use:
```python
#import libraries
from model_interference import AIModel
from skimage import io

#load model file
my_model = AIModel('/content/EfficientNetV2B0.h5')

#predict image from url
img = io.imread('https://i1.wp.com/allamerican1930.com/wp-content/uploads/2020/10/12oz-can-cluster.png?fit=1766,1434&ssl=1')
res = my_model.predict(img)
print(res)

img = io.imread('https://th.bing.com/th/id/OIP.L7qTmTqGbFQ53y4HrHqgSAHaHa?pid=ImgDet&rs=1')
res = my_model.predict(img)
print(res)

img = io.imread('https://th.bing.com/th/id/R.72f0d45317c34ab231625903b4670daa?rik=tF2VbhpfG9pBsg&pid=ImgRaw&r=0')
res = my_model.predict(img)
print(res)

img = io.imread('https://th.bing.com/th/id/OIP.J_yU0_Ut_HTPPpAQJzQ7AwHaEs?w=289&h=182&c=7&r=0&o=5&dpr=1.3&pid=1.7')
res = my_model.predict(img)
print(res)
```
Output:

```{'class': 'ALU', 'probability': 0.9884023}
{'class': 'PET', 'probability': 0.9809115}
{'class': 'MILKBOX', 'probability': 0.94315845}
{'class': 'OTHER', 'probability': 0.89215404}
```
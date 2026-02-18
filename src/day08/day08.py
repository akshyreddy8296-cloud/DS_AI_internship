import numpy as np
a=np.array([[1,2,3],[4,5,6]])
b=np.array([20,30,40])
c=np.array([[[1,2,3],[2,3,4],[4,5,6]]])
result=a+b
print(result)

#vectorization
arr=np.random.rand(100)
squared=arr**2
print(squared)

arr=np.arange(12)
reshaped=arr.reshape(3,4)
print(reshaped)

a=np.array([[1,2]])
b=np.array([[3,4]])

vstacked=np.vstack((a,b))
print(vstacked)

#statistical functions
data=np.array([[10,20,30],[40,50,60]])
print(np.mean(data))
print(np.mean(data,axis=0))

#the normalizer
import numpy as np

scores = np.random.randint(50, 101, size=(5, 3))

subject_mean = scores.mean(axis=0)

centered_scores = scores - subject_mean


print("Original Scores:\n", scores)
print("\nSubject-wise Mean:\n", subject_mean)
print("\nCentered Scores (After Broadcasting):\n", centered_scores)

#The Reshaper
import numpy as np

data = np.arange(24)

reshaped_data = data.reshape(4, 3, 2)

final_data = reshaped_data.transpose(0, 2, 1)

print("Final Shape:", final_data.shape)
print("Final Array:\n", final_data)
















#Introduction to pandas series

import pandas as pd

s1 = pd.Series([10, 20, 30, 40])
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

print(s1)
print(s2)

#Indexing and selection in series

marks = pd.Series([85, 90, 78], index=['Math', 'Physics', 'Chemistry'])

print(marks['Math'])
print(marks[['Math', 'Chemistry']])


#Boolean Masking in Series
scores = pd.Series([45, 67, 89, 34, 90])

passed = scores[scores > 60]
print(passed)

#Handling Missing data in series

data = pd.Series([10, None, 30, None])

print(data.isnull())
print(data.fillna(6.6))

#Vectorized String Operations
import pandas as pd

names = pd.Series(['Alice', 'bob', 'CHARLIE'])

print(names.str.lower())
print(names.str.contains('a'))

#Task 1:The product catalog
import pandas as pd
products=pd.Series([700,150,300],index=['Laptop','Mouse','keyboard'])

print(products['Laptop'])
print(products[['Laptop','Mouse']])
print(products)

#Task 2 The grade filter
import pandas as pd
grades=pd.Series([85,None,92,45,None,78,55])
missing_values=grades.isnull()
print(missing_values)
missing_values=grades.fillna(0)
print(missing_values)
scores=grades[grades>60]
print(scores)
print(grades)

#The Username Formatter
import pandas as pd
usernames=pd.Series(['Alice','Bob','Charli_data','daisy'])
print(usernames.str.strip())
print(usernames.str.lower())
print(usernames.str.contains("a"))
print(usernames)



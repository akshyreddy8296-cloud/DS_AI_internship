# import utlis
# print(utlis.multiply(2,4))

# import uttil
# print(uttil.add(2,4))

# Task 2:    (main.py)                                  
import math_operations

base = int(input("Enter base: "))
exp = int(input("Enter exponent: "))

nums = input("Enter numbers separated by space: ")
numbers_list = list(map(int, nums.split()))

p = math_operations.power(base, exp)
avg = math_operations.average(numbers_list)

print("Power result:", p)
print("Average:", avg)




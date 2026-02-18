import matplotlib.pyplot as plt
print("Task 1: Correlation Checker (Scatter Plots)")
study_hours = [1, 2, 3, 4, 5, 6, 7, 8]
scores = [50, 55, 65, 70, 75, 85, 90, 95]
plt.scatter(study_hours, scores)
plt.xlabel("Study_Hours")
plt.ylabel("Exam Scores")
plt.title("Study Hours vs Exam Scores")
plt.show()



print("Task 2: The Comparison Dashboard (Bar Charts & Subplots)")
categories = ['electronics', 'clothing','home']
sales = [300,450,200]

months = ['jan','feb','mar','apr','may']
trend_sales = [100,150,200,250,300]

plt.subplot(1,2,1)
plt.bar(categories,sales)
plt.title("sales by category")
plt.xlabel("product category")
plt.ylabel("sales")

plt.subplot(1,2,2)
plt.plot(months,trend_sales,marker = 'o')
plt.title('monthly sales trend')
plt.xlabel("month")
plt.ylabel("sales")
plt.tight_layout()
plt.show()



































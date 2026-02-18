
print("Task 1: The Trend Tracker (Line Plots)")
import matplotlib.pyplot as plt
months = [1, 2, 3, 4, 5]
revenue = [2000, 4500, 4000, 7500, 9000]
plt.plot(months, revenue, marker = "o" )
plt.title("Monthly Revenue Growth")
plt.xlabel("Month")
plt.ylabel("Revenue in USD")
plt.show()












print("Task 1: Correlation Checker (Scatter Plots)")
study_hours = [1, 2, 3, 4, 5, 6, 7, 8]
scores = [50, 55, 65, 70, 75, 85, 90, 95]
plt.scatter(study_hours, scores)
plt.xlabel("Study_Hours")
plt.ylabel("Exam Scores")
plt.title("Study Hours vs Exam Scores")
plt.show()






















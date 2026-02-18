#task 1  The personal logger
name = input("Enter your name: ")
daily_goal = input("Enter your Daily Goal: ")

# Open file in append mode
with open("journal.txt", "a") as file:
    file.write(f"Name: {name}, Daily Goal: {daily_goal}\n")

print("Your entry has been saved!")
file.close()

#The CSV student list

import csv

with open('students.csv','w', newline='') as csv_file:
    write = csv.writer(csv_file)
    write.writerow(['Name','Grade','Status'])
    write.writerow(['Alice','A','Pass'])
    write.writerow(['Bob','B','Pass'])
    write.writerow(['Charlie','F','Fail'])

with open('students.csv','r') as csv_read:
    reads = csv.reader(csv_read)
    for row in reads:
        if row[2] == "Pass":
            print(row[0])
        
        
#Task 03  the safe opener

filename = input("Enter the File Name (write only file name without Extension) : ")

try :
    file = open(f'{filename.txt}','r') 
    readers = file.read()
    print(readers)
except:
    print("Oops! That file doesn't exist yet")



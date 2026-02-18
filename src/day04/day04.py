# student={"name":"Hemanth","age":22,"course":"cse"}

# print(student["name"])
# print(student["age"])

# marks={"maths":80,"science":70,"english":80}
# for subject,score in marks.items():
#     print(subject,score)
    
# marks.update({"history":90})
# print(marks)

# purchases={
#     "alice":120,
#     "bob":86,
#     "charlie":67
# }

# for person,amount in purchases.items():
#     print(f"{person}: used ${amount}")

# n=int(input("Enter the no of customers : "))
# user_purchase={}
# for i in range(n):
#     name =input("enter customer name: ")
#     amount=int(input("enter purchase amount :"))
#     user_purchase[name]=amount
#     print("customer purchases data",user_purchases)
    
# top_customer=max(user_purchase,key=user_purchase.get) 
# print("Top customer:",top_customer)

# lowest_customer=min(user_purchase,key=user_purchase.get)
# print("lowest customer:",lowest_customer)

# a={1,2,3,4,}
# b={1,4,5}
# print(a|b)
# print(a&b)
# print(5 in a)
# print(5 in b)

# Task 1  personal contact book

# contacts={"hemanth":966,
#           "Akhil":977,
#           "avinash":955
#           }

# contacts.update({"akash":988})

# print(contacts)

# contacts.update({"hemanth":900})

# print(contacts)

# print("hemanth:",contacts.get("hemanth"))
# print("abhi:",contacts.get("abhi","contact not found"))

# for name, phone in contacts.items():
#     print(f"Contact:{name} , Phone:{phone}")

# Task 2  Duplicate cleaner sets

# raw_logs=["01","02","01","05","02","08","01"]
# unique_users=set(raw_logs)

# print(unique_users)

# print("05" in unique_users)

# print(len(raw_logs)-len(unique_users))


# Task 3 The Interest Matcher 


friend_a={"gaming","cooking","Hiking","Movies"}
friend_b={"gaming","Hiking","photography","Pyhton"}

print(friend_a & friend_b)
print(friend_a | friend_b)
print(friend_a - friend_b)













    
    






      
      
      
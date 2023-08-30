# A = [3, 1, 2, 2, 1, 4, 5]
# B = [1, 2, 2, 3, 3, 3, 4, 4, 5]

# # Create a sorting key function that uses the count of each item in B
# def sorting_key(item):
#     return B.count(item)

# # Sort list A using the custom sorting key
# A_sorted = sorted(A, key=sorting_key)

# print("Sorted A based on occurrences in B:", A_sorted)

list_ = [3, 2, 5, 7]

for i, item in enumerate(list_):
    print(i, item)
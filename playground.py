import cv2

def check_object_in_images(image1_path, image2_path, object_threshold=10):
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # Initialize Brute-Force Matcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)
    for match in matches:
        print(match[0].distance, match[1].distance)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Check if enough good matches exist
    if len(good_matches) >= object_threshold:
        print("The object is present in both images.")
    else:
        print("The object is not present in both images.")

    print(len(good_matches))

# Paths to your images
image1_path = 'save/misc/2mask.png'
image2_path = 'save/misc/5mask.png'

# Call the function
check_object_in_images(image1_path, image2_path)


# import cv2
# import numpy as np

# def feature_matching(image1, image2, object_threshold=10):    
#     # Initialize SIFT detector
#     sift = cv2.SIFT_create()

#     # Identify distinctive features in the first image.
#     keypoints1, descriptors1 = sift.detectAndCompute(image1, None)

#     # Identify corresponding features in the second image.
#     keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

#     # Match the features between the two images.
#     matcher = cv2.BFMatcher(cv2.NORM_L2)
#     matches = matcher.match(descriptors1, descriptors2)

#     # Sort the matches by their distance.
#     matches = sorted(matches, key=lambda x: x.distance)
#     for match in matches:
#         print(match.distance)

#     # # Keep only the best matches.
#     # good_matches = matches[:10]

#     # # Draw the matches on the images.
#     image_matches = None #cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)

#     # Apply ratio test to get good matches
#     # good_matches = []
#     # for m, n in matches:
#     #     if m.distance < 0.75 * n.distance:
#     #         good_matches.append(m)

#     # Check if enough good matches exist
#     if len(matches) >= object_threshold:
#         print("The object is present in both images.")
#     else:
#         print("The object is not present in both images.")

#     # print(len(good_matches))

#     return image_matches

# # Paths to your images
# image1_path = 'save/misc/target_mask.png'
# image2_path = 'save/misc/target_mask.png'

# image1 = cv2.imread(image1_path)
# image2 = cv2.imread(image2_path)

# # Check if the object appears in both images using feature matching.
# image_matches = feature_matching(image1, image2)
# # cv2.imshow("Feature matching", image_matches)
# # cv2.waitKey(0)

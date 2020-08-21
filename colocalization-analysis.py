#upload and import needed libraries
import PIL
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time

#NEW!both channels opened as matricies
def colocalization_analysis (image_name, optimal_threshold, filter_min_size, filter_max_size):
    start_time = time.time()
    #specify size filter
    s1 = filter_min_size
    s2 = filter_max_size

    #specify threshold, consider implementing adaprive thresholding if the image is too heterogeneous
    thresh = optimal_threshold

    #read the image, split and save as array
    image = Image.open(image_name).convert("RGB")
    R,G,B = image.split()
    R.save("R.jpg")
    G.save("G.jpg")
    B.save("B.jpg")
    # R = Image.open('result.jpg')
    # G = Image.open('resultG.jpg')
    red_channel_array = np.asarray(R)
    green_channel_array = np.asarray(G)

    #threshold all channel and save thresholded images
    ret, red_channel_thresholded = cv2.threshold(red_channel_array, thresh, 255, cv2.THRESH_BINARY)
    ret, green_channel_thresholded = cv2.threshold(green_channel_array, thresh, 255, cv2.THRESH_BINARY)

    thresholded_red = Image.fromarray(red_channel_thresholded)
    thresholded_red.save("Thresholded_R.jpg")

    thresholded_green = Image.fromarray(green_channel_thresholded)
    thresholded_green.save("Thresholded_G.jpg")


    #find all contours on the RED channel, draw all these contours, calculate their area and count them. Print the output
    red_contours, hierarchy = cv2.findContours(red_channel_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    selected_red_contours = []
    for i in red_contours:
        area = cv2.contourArea(i)
        if s1 < area < s2:
            areas.append(area)
            selected_red_contours.append(i)
    objects_number = len(areas)
    print(f"\nNumber of red signal: " + str(objects_number))

    #find all contours on the GREEN channel, draw all these contours, calculate their area and count them. Print the output

    green_contours, hierarchy = cv2.findContours(green_channel_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas2 = []
    selected_green_contours = []
    for i in green_contours:
        area = cv2.contourArea(i)
        if s1 < area < s2:
            areas2.append(area)
            selected_green_contours.append(i)

    green_objects_number = len(areas2)
    print(f"\nNumber of green signal: " + str(green_objects_number))

    #show contours for the red channel
    red_original = cv2.imread("R.jpg")
    for contour in selected_red_contours:
        color_gen = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        red_results = cv2.drawContours(red_original, contour, -1, (255,255,0), 7)
    x = Image.fromarray(red_results, "RGB")
    x.save("red_results.jpg")

    #same on the blank image
    height, width = red_channel_thresholded.shape
    blank_im = np.zeros((height, width, 3), np.uint8)
    for contour in selected_red_contours:
        color_gen = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        red_results_on_blank = cv2.drawContours(blank_im, contour, -1, color_gen, 7)
    x = Image.fromarray(red_results_on_blank, "RGB")
    x.save("red_results_on_blank.jpg")

    #show contours for the green channel
    green_original = cv2.imread("G.jpg")
    for contour in selected_green_contours:
        color_gen = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        green_results = cv2.drawContours(green_original, contour, -1, (255,0,255), 7)
    y = Image.fromarray(green_results, "RGB")
    y.save("green_results.jpg")

    #same on the blank 
    height, width = red_channel_thresholded.shape
    blank_im = np.zeros((height, width, 3), np.uint8)
    for contour in selected_green_contours:
        color_gen = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        green_results_on_blank = cv2.drawContours(blank_im, contour, -1, color_gen, 7)
    y = Image.fromarray(green_results_on_blank, "RGB")
    y.save("green_results_on_blank.jpg")

    #find intersections
    height, width = red_channel_thresholded.shape
    blank_im = np.zeros((height, width, 3), np.uint8)
    intersections = np.logical_and(red_results_on_blank, green_results_on_blank, blank_im)
    intersections[intersections > 0] = 255
    z = Image.fromarray(intersections)
    R,G,B = z.split()
    B.save("intersections.jpg")
    intersections_array = np.asarray(B)

    #to represent well the intersecting points - create contours of intersections and draw them directly on the original image
    im = cv2.imread(image_name, 1)
    intersection_contours, hierarchy = cv2.findContours(intersections_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in intersection_contours:
        count = count + 1
        color_gen = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        image = cv2.drawContours(im, contour, -1, color_gen, 7)
    intersections_on_the_image = Image.fromarray(image)
    intersections_on_the_image.save("Intersections_on_the_image.jpg")

    #find area of intersecting objects if area is too big, count as 2 or 3 objects.
    areas3 = []
    selected_intersection_contours = []
    for i in intersection_contours:
        area = cv2.contourArea(i)
        if s1 < area < s2:
            areas3.append(area)
            selected_intersection_contours.append(i)
    selected_intersection_contours_number = len(areas3)
    print(f"\nNumber of filtered_intersections: " + str(selected_intersection_contours_number))
    print(f"\nNumber of intersections: " + str(count))
    return count, objects_number, green_objects_number
    print("--- %s seconds ---" % (time.time() - start_time))

colocalization_analysis("combined 2 red+green.jpg", 120, 1, 180)

# # threshold experiment, here I experiment with thresholding to find the optimal value. 
# thresholds = [50,75,90,105,120,135,150,165,170,185]
# red_signal_for_each_threshold = []
# green_signal_for_each_threshold = []
# intersections_for_each_threshold = []
# for threshold in thresholds:
#     count, objects_number, green_objects_number = colocalization_analysis("combined 2 red+green.jpg", threshold, 1, 180)
#     intersections_for_each_threshold.append(count)
#     red_signal_for_each_threshold.append(objects_number)
#     green_signal_for_each_threshold.append(green_objects_number)
# xnumbers = np.linspace(0,255,10)
# ynumbers = np.linspace(0,1500,20)
# plt.plot(thresholds,intersections_for_each_threshold, color = 'b', label = 'Intersections found')
# plt.plot(thresholds,red_signal_for_each_threshold, color = 'r', label = 'Red found')
# plt.plot(thresholds,green_signal_for_each_threshold, color = 'g', label = 'Green found')
# plt.xlabel("Thresholds values")
# plt.ylabel("Amount of objects detected")
# plt.title("Threshold experiment with RB cleaning")
# plt.xticks(xnumbers)
# plt.yticks(ynumbers)
# plt.legend()
# plt.axis([0, 255, 0, 1500])
# plt.show()


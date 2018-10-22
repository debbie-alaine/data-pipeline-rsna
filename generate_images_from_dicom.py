import pydicom
import csv
from collections import defaultdict
import numpy as np
import shutil
from matplotlib import pyplot
import random
from scipy import ndimage, misc
from functools import partial
import os
import warnings
import json

def create_bounding_box_map(filepath):
    box_map = defaultdict(lambda: [])

    with open(filepath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        # skip headers
        next(csv_reader, None)
        for row in csv_reader:
            patient_id = row[0]
            x = row[1]
            y = row[2]
            width = row[3]
            height = row[4]
            target = row[5]

            if int(target):
                box_map[patient_id].append([int(float(a)) for a in [x, y, width, height]])

        ## Print Map
        # for patient_id in box_map:
        #     print(patient_id + ": " + str(box_map[patient_id]))

        return box_map


def read_dicom_files(dicom_path):
    return pydicom.dcmread(dicom_path)


def dicom_to_array(ds):
    image_array = np.zeros([1024, 1024], dtype=ds.pixel_array.dtype)
    image_array[:, :] = ds.pixel_array
    return image_array


def plot_image_and_bounding_boxes(image_array, boxes):
    pyplot.set_cmap(pyplot.gray())
    pyplot.imshow(image_array)
    for [x, y, w, h] in boxes:
        pyplot.plot([x, x, x + w, x + w, x], [y, y + h, y + h, y, y])
    pyplot.show()


# shift image randomly and replace displacement with black
def shift_image(x, y, image_array, box_list):
    rx = random.randint(-x, x)
    ry = random.randint(-y, y)
    shift_image_box_list = []

    shifted_array = np.zeros((1024, 1024))
    ndimage.shift(image_array, (ry, rx), shifted_array, mode="constant", cval=0)

    for [x, y, w, h] in box_list:
        shift_image_box_list.append([x + rx, y + ry, w, h])

    return shifted_array, shift_image_box_list


def flip_image(image_array, box_list):
    copy_array = np.copy(image_array)
    flipped_box_list = []

    for [x, y, w, h] in box_list:
        flipped_box_list.append([1024 - x - w, y, w, h])

    return np.fliplr(copy_array), flipped_box_list


def shift_bbox(x, y, image_array, box_list):
    image_copy = np.copy(image_array)
    shifted_bbox_list = []

    def inside(x, y, w, h, px, py):
        return x <= px < x + w and y <= py < y + h

    for (idx, [x0, y0, w, h]) in enumerate(box_list, 0):
        rx = random.randint(-x, x)
        ry = random.randint(-y, y)

        while (y0 + ry < 0) or (x0 + rx < 0):
            rx = random.randint(-x, x)
            ry = random.randint(-y, y)

        inside_fns = [partial(inside, x1, y1, w1, h1) for (idx2, [x1, y1, w1, h1]) in enumerate(box_list) if
                      idx != idx2]
        if np.any(
                [np.any([inside_fn(x0 + rx, y0 + ry), inside_fn(x0 + w + rx, y0 + ry), inside_fn(x0 + rx, y0 + h + ry),
                         inside_fn(x0 + w + rx, y0 + h + ry)]) for inside_fn in inside_fns]):
            continue
        box = np.copy(image_copy[y0:y0 + h, x0:x0 + w])
        image_copy[y0:y0 + h, x0:x0 + w].fill(0)

        image_copy[(y0 + ry):(y0 + h + ry), (x0 + rx):(x0 + w + rx)] = box

        shifted_bbox_list.append([x0 + rx, y0 + ry, w, h])
    return image_copy, shifted_bbox_list


def scale_bbox(factor, image_array, box_list):
    image_copy = np.copy(image_array)
    scaled_bbox_list = []

    for (index, [x, y, w, h]) in enumerate(box_list):

        rf = random.uniform(1.0 / (1.0 + factor), 1.0 + factor)
        new_y = y + round(h * rf)
        new_x = x + round(w * rf)

        while (new_y > 1024) or (new_x > 1024):
            rf = random.uniform(1.0 / (1.0 + factor), 1.0 + factor)
            new_y = y + round(h * rf)
            new_x = x + round(w * rf)

        box = np.copy(image_copy[y:y + h, x:x + w])
        image_copy[y:(y + h), x:(x + w)].fill(0)

        # new_box = np.zeros([round(h * rf), round(w * rf)], dtype=int)
        new_box = ndimage.zoom(box, rf, mode='nearest')

        center = (y + round(h / 2), x + round(w / 2))
        nx = center[1] - round((w * rf) / 2)
        ny = center[0] - round((h * rf) / 2)

        # check if bounding boxes are out of bounds
        if ny < 0:
            ny = 0
        if nx < 0:
            nx = 0

        # image_copy[ny:(ny + round(h * rf)), nx:(nx + round(w * rf))] = new_box
        image_copy[ny:(ny + new_box.shape[0]), nx:(nx + new_box.shape[1])] = new_box

        scaled_bbox_list.append([nx, ny, round(w * rf), round(h * rf)])
    return image_copy, scaled_bbox_list


def scale_image(factor, image_array, box_list):
    rf = random.uniform(1.0 / (1.0 + factor), 1.0 + factor)
    output = np.zeros([1024, 1024], dtype=int)
    zoom_output = np.zeros([round(1024 * rf), round(1024 * rf)], dtype=int)
    ndimage.zoom(image_array, rf, zoom_output, mode="nearest")

    new_box_list = []

    # if the image is shrinking
    if rf < 1:
        lower = 512 - round(zoom_output.shape[0] / 2)
        upper = 512 + round(zoom_output.shape[0] / 2)

        if output[lower:upper, lower:upper].shape[0] > zoom_output.shape[0]:
            lower = lower + 1
        if output[lower:upper, lower:upper].shape[0] < zoom_output.shape[0]:
            lower = lower - 1

        output[lower:upper, lower:upper] = zoom_output

    # if the image is growing
    if rf > 1:
        lower = round(zoom_output.shape[0] / 2) - 512
        upper = round(zoom_output.shape[0] / 2) + 512

        if zoom_output[lower:upper, lower:upper].shape[0] > 1024:
            lower = lower + 1
        if zoom_output[lower:upper, lower:upper].shape[0] < 1024:
            lower = lower - 1

        if lower < 0:
            upper = 0

        output = zoom_output[lower:upper, lower:upper]

    for [x, y, w, h] in box_list:
        c1 = (round(rf * (x - 512) + 512), round(rf * (y - 512)) + 512)
        c2 = (round(rf * (x + w - 512) + 512), round(rf * (y - 512)) + 512)
        c3 = (round(rf * (x - 512) + 512), round(rf * (y + h - 512)) + 512)
        c4 = (round(rf * (x + w - 512) + 512), round(rf * (y + h - 512)) + 512)
        new_box_list.append([c1[0], c1[1], c2[0] - c1[0], c3[1] - c1[1]])

    return output, new_box_list


if __name__ == "__main__":
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    training_images_dir = "./training_data_test"
    training_bounding_boxes = "./stage_1_train_labels.csv"

    generated_image_dir = "./generated_images"

    if os.path.exists(generated_image_dir):
        shutil.rmtree(generated_image_dir)

    os.mkdir(generated_image_dir)

    # create dictionary of bounding boxes for each patient id
    box_map = create_bounding_box_map(training_bounding_boxes)
    print("Creating dictionary of bounding boxes...\n")

    # new box map for generated images
    final_box_map = defaultdict(lambda: [])

    ###################
    ### SHIFT IMAGE ###
    ###################
    shift_image_dir = "./generated_images/shift_image"

    if os.path.exists(shift_image_dir):
        shutil.rmtree(shift_image_dir)

    os.mkdir(shift_image_dir)

    print("Running shift image operation: ")
    for file in os.listdir(training_images_dir):
        if file.endswith(".dcm"):
            patient_id = file.replace(".dcm", "")
            # read dicom file and convert to numpy array
            array = dicom_to_array(read_dicom_files("{}/{}".format(training_images_dir, file)))

            for x in range(0, 5):
                shifted_array, shifted_box_list = shift_image(10, 10, array, box_map[patient_id])
                shifted_patient_id = "{}-shift-{}".format(patient_id, x)
                misc.imsave("{}/{}.png".format(shift_image_dir, shifted_patient_id), shifted_array)
                final_box_map[shifted_patient_id] = shifted_box_list
                # plot_image_and_bounding_boxes(shifted_array, shifted_box_list)
                # flip shifted image and save
                flipped_shift_array, flipped_shift_box_list = flip_image(shifted_array, shifted_box_list)
                flipped_patient_id = "{}-shift-flipped-{}".format(patient_id, x)
                misc.imsave("{}/{}.png".format(shift_image_dir, flipped_patient_id), flipped_shift_array)
                final_box_map[flipped_patient_id] = flipped_shift_box_list


    images_generated = len([name for name in os.listdir(shift_image_dir)])
    print("{} images generated in {}\n".format(images_generated, shift_image_dir))

    ##########################
    ### SHIFT BOUNDING BOX ###
    ##########################
    shift_bbox_dir = "./generated_images/shift_bbox"

    if os.path.exists(shift_bbox_dir):
        shutil.rmtree(shift_bbox_dir)

    os.mkdir(shift_bbox_dir)

    print("Running shift bounding box operation: ")
    for file in os.listdir(training_images_dir):
        if file.endswith(".dcm"):
            patient_id = file.replace(".dcm", "")
            # check that bounding boxes exist for image
            if box_map[patient_id]:
                # read dicom file and convert to numpy array
                array = dicom_to_array(read_dicom_files("{}/{}".format(training_images_dir, file)))

                if box_map[patient_id]:
                    for x in range(0, 25):
                        shifted_bbox_array, shifted_bbox_box_list = shift_bbox(50, 50, array, box_map[patient_id])
                        shifted_bbox_patient_id = "{}-shift-bbox-{}".format(patient_id, x)
                        misc.imsave("{}/{}.png".format(shift_bbox_dir, shifted_bbox_patient_id), shifted_bbox_array)
                        final_box_map[shifted_bbox_patient_id] = shifted_bbox_box_list
                        # plot_image_and_bounding_boxes(shifted_bbox_array, shifted_bbox_box_list)
                        # flip image and save
                        flipped_bbox_shift_array, flipped_bbox_shift_box_list = flip_image(shifted_bbox_array,
                                                                                           shifted_bbox_box_list)
                        flipped_bbox_patient_id = "{}-shift-bbox-flipped-{}".format(patient_id, x)
                        misc.imsave("{}/{}.png".format(shift_bbox_dir, flipped_bbox_patient_id), flipped_bbox_shift_array)
                        final_box_map[flipped_bbox_patient_id] = flipped_bbox_shift_box_list

    images_generated = len([name for name in os.listdir(shift_bbox_dir)])
    print("{} images generated in {}\n".format(images_generated, shift_bbox_dir))

    ##########################
    ### SCALE BOUNDING BOX ###
    ##########################
    scale_bbox_dir = "./generated_images/scale_bbox"

    if os.path.exists(scale_bbox_dir):
        shutil.rmtree(scale_bbox_dir)

    os.mkdir(scale_bbox_dir)

    print("Running scale bounding box operation: ")
    for file in os.listdir(training_images_dir):
        if file.endswith(".dcm"):
            patient_id = file.replace(".dcm", "")
            # read dicom file and convert to numpy array
            array = dicom_to_array(read_dicom_files("{}/{}".format(training_images_dir, file)))
            if box_map[patient_id]:
                for x in range(0, 25):
                    scale_bbox_array, scale_bbox_box_list = scale_bbox(.25, array, box_map[patient_id])
                    scale_bbox_patient_id = "{}-scale-bbox-{}".format(patient_id, x)
                    misc.imsave("{}/{}.png".format(scale_bbox_dir, scale_bbox_patient_id), scale_bbox_array)
                    final_box_map[scale_bbox_patient_id] = scale_bbox_box_list
                    # plot_image_and_bounding_boxes(scale_bbox_array, scale_bbox_box_list)
                    # flip image and save
                    flipped_scale_bbox_array, flipped_scale_bbox_box_list = flip_image(scale_bbox_array,
                                                                                       scale_bbox_box_list)
                    flipped_scale_bbox_patient_id = "{}-scale-bbox-flipped-{}".format(patient_id, x)
                    misc.imsave("{}/{}.png".format(scale_bbox_dir, flipped_scale_bbox_patient_id),
                                flipped_scale_bbox_array)
                    final_box_map[flipped_scale_bbox_patient_id] = flipped_scale_bbox_box_list

    images_generated = len([name for name in os.listdir(scale_bbox_dir)])
    print("{} images generated in {}\n".format(images_generated, scale_bbox_dir))

    ##################
    ## SCALE IMAGE ###
    ##################
    scale_image_dir = "./generated_images/scale_image"

    if os.path.exists(scale_image_dir):
        shutil.rmtree(scale_image_dir)

    os.mkdir(scale_image_dir)

    print("Running scale image operation: ")
    for file in os.listdir(training_images_dir):
        if file.endswith(".dcm"):
            patient_id = file.replace(".dcm", "")
            # read dicom file and convert to numpy array
            array = dicom_to_array(read_dicom_files("{}/{}".format(training_images_dir, file)))

            for x in range(0, 5):
                scale_image_array, scale_image_box_list = scale_image(.0625, array, box_map[patient_id])
                scale_image_patient_id = "{}-scale-image-{}".format(patient_id, x)
                misc.imsave("{}/{}.png".format(scale_image_dir, scale_image_patient_id), scale_image_array)
                final_box_map[scale_image_patient_id] = scale_image_box_list
                # plot_image_and_bounding_boxes(scale_image_array, scale_image_box_list)
                # flip image and save
                flipped_scale_image_array, flipped_scale_image_box_list = flip_image(scale_image_array,
                                                                                     scale_image_box_list)
                flipped_scale_image_patient_id = "{}-scale-image-flipped-{}".format(patient_id, x)
                misc.imsave("{}/{}.png".format(scale_image_dir, flipped_scale_image_patient_id),
                            flipped_scale_image_array)
                final_box_map[flipped_scale_image_patient_id] = flipped_scale_image_box_list

    images_generated = len([name for name in os.listdir(scale_image_dir)])
    print("{} images generated in {}\n".format(images_generated, scale_image_dir))


    #########################
    ## SCALE & SHIFT BBOXES #
    #########################

    scale_shift_bbox_dir = "./generated_images/scale_shift_bbox"

    if os.path.exists(scale_shift_bbox_dir):
        shutil.rmtree(scale_shift_bbox_dir)

    os.mkdir(scale_shift_bbox_dir)

    print("Running scale and shift bounding box operations: ")
    for file in os.listdir(training_images_dir):
        if file.endswith(".dcm"):
            patient_id = file.replace(".dcm", "")
            # read dicom file and convert to numpy array
            array = dicom_to_array(read_dicom_files("{}/{}".format(training_images_dir, file)))
            if box_map[patient_id]:
                for x in range(0, 25):
                    scale_bbox_array, scale_bbox_box_list = scale_bbox(.25, array, box_map[patient_id])
                    scale_shift_bbox_array, scale_shift_bbox_box_list = shift_bbox(50, 50, scale_bbox_array, scale_bbox_box_list);
                    scale_shift_bbox_patient_id = "{}-scale-shift-bbox-{}".format(patient_id, x)
                    misc.imsave("{}/{}.png".format(scale_shift_bbox_dir, scale_shift_bbox_patient_id), scale_shift_bbox_array)
                    final_box_map[scale_shift_bbox_patient_id] = scale_bbox_box_list
                    # plot_image_and_bounding_boxes(scale_shift_bbox_array, scale_shift_bbox_box_list)
                    # flip image and save
                    flipped_scale_bbox_array, flipped_scale_shift_box_box_list = flip_image(scale_shift_bbox_array,
                                                                                       scale_shift_bbox_box_list)
                    flipped_scale_shift_bbox_patient_id = "{}-scale-shift-bbox-flipped-{}".format(patient_id, x)
                    misc.imsave("{}/{}.png".format(scale_shift_bbox_dir, flipped_scale_shift_bbox_patient_id),
                                flipped_scale_bbox_array)
                    final_box_map[flipped_scale_shift_bbox_patient_id] = flipped_scale_shift_box_box_list

    images_generated = len([name for name in os.listdir(scale_shift_bbox_dir)])
    print("{} images generated in {}\n".format(images_generated, scale_shift_bbox_dir))

    ################################
    ## SHIFT IMAGE & SHIFT BBOXES ##
    ################################

    shift_image_shift_bbox_dir = "./generated_images/shift_image_shift_bbox"

    if os.path.exists(shift_image_shift_bbox_dir):
        shutil.rmtree(shift_image_shift_bbox_dir)

    os.mkdir(shift_image_shift_bbox_dir)

    print("Running shift image and shift bbox operation: ")
    for file in os.listdir(training_images_dir):
        if file.endswith(".dcm"):
            patient_id = file.replace(".dcm", "")
            # read dicom file and convert to numpy array
            array = dicom_to_array(read_dicom_files("{}/{}".format(training_images_dir, file)))

            if box_map[patient_id]:
                for x in range(0, 5):
                    shifted_array, shifted_box_list = shift_image(10, 10, array, box_map[patient_id])
                    shift_shift_array, shift_shift_box_list = shift_bbox(50, 50, shifted_array, shifted_box_list)
                    shift_shift_patient_id = "{}-shift-shift-{}".format(patient_id, x)
                    misc.imsave("{}/{}.png".format(shift_image_shift_bbox_dir, shift_shift_patient_id), shift_shift_array)
                    final_box_map[shift_shift_patient_id] = shift_shift_box_list
                    # plot_image_and_bounding_boxes(shift_shift_array, shift_shift_box_list)
                    # flip shifted image and save
                    flipped_shift_shift_array, flipped_shift_shift_box_list = flip_image(shift_shift_array, shift_shift_box_list)
                    flipped_patient_id = "{}-shift-shift-flipped-{}".format(patient_id, x)
                    misc.imsave("{}/{}.png".format(shift_image_shift_bbox_dir, flipped_patient_id), flipped_shift_shift_array)
                    final_box_map[flipped_patient_id] = flipped_shift_shift_box_list

    images_generated = len([name for name in os.listdir(shift_image_shift_bbox_dir)])
    print("{} images generated in {}\n".format(images_generated, shift_image_shift_bbox_dir))

    ########################################
    ## SCALE IMAGE & SHIFT AND SCALE BBOX ##
    ########################################

    scale_image_scale_shift_bbox_dir = "./generated_images/scale_image_scale_shift_bbox"

    if os.path.exists(scale_image_scale_shift_bbox_dir):
        shutil.rmtree(scale_image_scale_shift_bbox_dir)

    os.mkdir(scale_image_scale_shift_bbox_dir)

    print("Running scale image and scale and shift bounding box operations: ")
    for file in os.listdir(training_images_dir):
        if file.endswith(".dcm"):
            patient_id = file.replace(".dcm", "")
            # read dicom file and convert to numpy array
            array = dicom_to_array(read_dicom_files("{}/{}".format(training_images_dir, file)))
            if box_map[patient_id]:
                for x in range(0, 5):
                    scale_image_array, scale_image_bbox_list = scale_image(0.625, array, box_map[patient_id])
                    scale_scale_bbox_array, scale_scale_bbox_list = scale_bbox(.25, scale_image_array, scale_image_bbox_list)
                    scale_scale_shift_bbox_array, scale_scale_shift_bbox_list = shift_bbox(50, 50, scale_scale_bbox_array, scale_scale_bbox_list)
                    scale_scale_shift_bbox_patient_id = "{}-scale-scale-shift-bbox-{}".format(patient_id, x)
                    misc.imsave("{}/{}.png".format(scale_image_scale_shift_bbox_dir, scale_scale_shift_bbox_patient_id), scale_scale_shift_bbox_array)
                    final_box_map[scale_scale_shift_bbox_patient_id] = scale_scale_shift_bbox_list
                    # plot_image_and_bounding_boxes(scale_scale_shift_bbox_array, scale_scale_shift_bbox_list)
                    # flip image and save
                    flipped_scale_bbox_array, flipped_scale_shift_box_box_list = flip_image(scale_scale_shift_bbox_array,
                                                                                            scale_scale_shift_bbox_list)
                    flipped_scale_shift_bbox_patient_id = "{}-scale-scale-shift-bbox-flipped-{}".format(patient_id, x)
                    misc.imsave("{}/{}.png".format(scale_image_scale_shift_bbox_dir, flipped_scale_shift_bbox_patient_id),
                                flipped_scale_bbox_array)
                    final_box_map[flipped_scale_shift_bbox_patient_id] = flipped_scale_shift_box_box_list

    images_generated = len([name for name in os.listdir(scale_image_scale_shift_bbox_dir)])
    print("{} images generated in {}\n".format(images_generated, scale_image_scale_shift_bbox_dir))

    generated_box_map_path = "./generated_box_map.json"
    if os.path.exists(generated_box_map_path):
        os.remove(generated_box_map_path)

    with open('generated_box_map.json', 'w') as outfile:
        json.dump(final_box_map, outfile)

    print("IMAGE GENERATION COMPLETE!!!")
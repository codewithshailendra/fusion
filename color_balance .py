# step 1 Input image: this is underwater image.
# step 2 Apply white balance to the input image, the obtained image is the first input of the fusion process, suppose the obtained image is img1.
# step 3 Apply temporal coherent noise reduction method to img1 to obtain another input of fusion image, suppose this image is img2, which is obtained by local adaptive histogram equalization
# step 4 Obtain the weights of these two inputs. 4 types of weights are to be found for both of these images.
#       Laplacian contrast weight
#      Local Contrast weight
#      Saliency weight
#      Exposedness weight

# step 5 Finally, the multi-scale fusion process is applied to generate the restored image. This step consists of following step:
#   Gaussian pyramid decomposition for weight maps.
#   Laplacian pyramid decomposition
#   pyramid reconstruct
#   FilterMask

# step 6 Fusion process


# color balancing using histogram normalisation.
# colourful image  and the percentage of pixels to clip to white and black(normally (1,10)).
# this functions returns the color_balanced image.

from PIL import Image
import cv2
import numpy as np


def color_balance(img, percent):
    if percent <= 0:
        percent = 5  # taken as an average of (1-10).

    rows = img.shape[0]
    cols = img.shape[1]
    # knowing the no. of channels in the present image
    no_of_chanl = img.shape[2]

    # halving the given percentage based on the given research paper
    halfpercent = percent/200.0

    # list for storing all the present channels of the image separately.
    channels = []

    if no_of_chanl == 3:
        for i in range(3):
            # add all the present channels of the image to this list separately
            channels.append(img[:, :, i:i+1])
    else:
        channels.append(img)

    results = []

    for i in range(no_of_chanl):
        #print(channels[i].shape)
        plane = channels[i].reshape(1, rows*cols, 1)
        plane.sort()
        lower_value = plane[0][int(plane.shape[1]*halfpercent)][0]
        top_value = plane[0][int(plane.shape[1]*(1-halfpercent))][0]

        channel = channels[i]

        for p in range(rows):
            for q in range(cols):
                if channel[p][q][0] < lower_value:
                    channel[p][q][0] = lower_value
                if channel[p][q][0] < top_value:
                    channel[p][q][0] = top_value

        channel = cv2.normalize(channel, None, 0.0, 255.0/2, cv2.NORM_MINMAX)
        # convert the image in desired format-converted

        results.append(channel)

    output_image = np.zeros((rows, cols, 3))
    for x in results:
        cv2.imshow('image', x)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    output_image = cv2.merge(results)
    return output_image



if __name__ == "__main__":
    img = cv2.imread("G:/54429358_369579283646633_9215797332648625036_n.jpg")
    cv2.imshow('original_image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(type(img))
    percent = 100.0
    final_image = color_balance(img, percent)
    cv2.imshow('image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    




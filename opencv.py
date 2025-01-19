import os
import numpy as np
import cv2
from PIL import Image


class ImageProcessor:
    def __init__(self, mask_folder, ori_folder, output_folder=None):
        self.mask_folder = mask_folder
        self.ori_folder = ori_folder
        self.output_folder = output_folder or os.path.join(
            os.path.dirname(self.ori_folder), "output")
        os.makedirs(self.output_folder, exist_ok=True)

    def process_and_merge_images(self, mask_files, ori_files):
        for mask_filename, ori_filename in zip(mask_files, ori_files):
            try:

                mask_image_path = os.path.join(self.mask_folder, mask_filename)
                mask_image = self.load_and_process_mask(mask_image_path)

                ori_image_path = os.path.join(self.ori_folder, ori_filename)
                ori_image = self.load_and_resize_image(
                    ori_image_path, mask_image.shape[1], mask_image.shape[0])

                result_image = cv2.bitwise_or(ori_image, mask_image)

                output_image_path = os.path.join(
                    self.output_folder, os.path.splitext(ori_filename)[0] + ".png")
                cv2.imwrite(output_image_path, result_image)

            except Exception as e:
                print(
                    f"Error processing {ori_filename} and {mask_filename}: {e}")

    def load_and_process_mask(self, mask_image_path):
        tiff = Image.open(mask_image_path)
        tiff_array = np.array(tiff)

        # Normalize and convert to uint8
        img = (np.maximum(tiff_array, 0) / tiff_array.max()) * 255.0
        img_uint8 = img.astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        # Create binary mask and find contours
        _, binary_mask = cv2.threshold(img_uint8, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Draw contours and bounding boxes on mask
        mask = np.zeros_like(img_bgr)
        cv2.drawContours(mask, contours, -1, (0, 255, 0), 4)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 160, 0), 4)

        return mask

    def load_and_resize_image(self, image_path, width, height):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to load image at {image_path}")
        return cv2.resize(image, (width, height))


def main():
    mask_folder = "D:/detect_0506/EfficientAD-main/output1/1/anomaly_maps/mvtec_ad/rj45/test/broken"
    ori_folder = "D:/detect_0506/EfficientAD-main/mvtec_anomaly_detection/rj45/test/broken"

    mask_files = sorted(os.listdir(mask_folder))
    ori_files = sorted(os.listdir(ori_folder))

    processor = ImageProcessor(mask_folder, ori_folder)
    processor.process_and_merge_images(mask_files, ori_files)


if __name__ == "__main__":
    main()


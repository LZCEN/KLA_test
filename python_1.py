import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image, ImageEnhance

def process_image(image_path):
    # Load image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_array = np.array(image)

    # Extract X, Y direction pixel number
    y_pixels, x_pixels = image_array.shape
    print(f'X direction pixel number: {x_pixels}')
    print(f'Y direction pixel number: {y_pixels}')

    # Plot grayscale histogram
    plt.figure(figsize=(10, 5))
    plt.hist(image_array.flatten(), bins=256, range=(0, 255), density=True, color='gray', alpha=0.7)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig('Grayscale Histogram')
    plt.close()

    # Plot X direction average grayscale
    x_avg_grayscale = np.mean(image_array, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(x_avg_grayscale, color='black')
    plt.title('X Direction Average Grayscale')
    plt.xlabel('X Pixel Index')
    plt.ylabel('Average Grayscale Value')
    plt.savefig('X Direction Average Grayscale.jpg')
    plt.close()

    # Plot Y direction average grayscale
    y_avg_grayscale = np.mean(image_array, axis=1)
    plt.figure(figsize=(10, 5))
    plt.plot(y_avg_grayscale, color='black')
    plt.title('Y Direction Average Grayscale')
    plt.xlabel('Y Pixel Index')
    plt.ylabel('Average Grayscale Value')
    plt.savefig('Y Direction Average Grayscale.jpg')
    plt.close()
    

def extract_color_pixels(image_path, target_color, tolerance=100):

    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    target_color = np.array(target_color)

    mask = np.all(np.abs(image_array - target_color) <= tolerance, axis=-1)
    color_pixels = np.argwhere(mask)

    x_indices = color_pixels[:, 1]

    pixel = np.max(x_indices) - np.min(x_indices)
    pixel_size = 1000/pixel
    return pixel_size


image_path = 'image.jpg' 
target_color = (255,255,0)
process_image(image_path)
pixel_size = extract_color_pixels(image_path, target_color, tolerance=100)
print("Pixel size (nm):", pixel_size)

# For extract the structures
def find_local_minima(array):

    local_minima = []
    for i in range(1, len(array) - 1):
        if array[i - 1] > array[i] < array[i + 1]:
            local_minima.append(i)
    
    return local_minima

def calculate_distances(local_minima):
    distances = []
    for i in range(len(local_minima) - 1):
        distances.append(local_minima[i + 1] - local_minima[i])
    return distances


def crop_and_merge(image_path, y_ranges, output_path,factor, output_path_2):
    # Load image
    image = Image.open(image_path).convert('L')
    width, height = image.size
    cropped_images = []
   
    for i in range(0,len(y_ranges) - 1,2):
        y_start = y_ranges[i]
        y_end = y_ranges[i + 1]
        
        box = (0, y_start, width, y_end)
        cropped_image = image.crop(box)
        cropped_images.append(cropped_image)
    
    if cropped_images:
        
        total_height = sum(img.height for img in cropped_images)
        max_width = max(img.width for img in cropped_images)
        merged_image = Image.new('RGB', (max_width, total_height))
        
        y_offset = 0
        for img in cropped_images:
            merged_image.paste(img, (0, y_offset))
            y_offset += img.height

        merged_image.save(output_path)
        print(f'Saved {output_path}')
    else:
        print('Fail')
    
    enhancer = ImageEnhance.Brightness(merged_image)
    darker_image = enhancer.enhance(factor)
    darker_image.save(output_path_2)



image = Image.open(image_path).convert('L')
image_array = np.array(image)
y_avg_grayscale = np.mean(image_array, axis=1)
array = y_avg_grayscale
local_minima = find_local_minima(array)
distances = calculate_distances(local_minima) # to choose the range for capture region of interest
indices = np.arange(1, len(local_minima), 5)
values = []
for index in indices:
        if index < len(local_minima): 
            values.append(local_minima[index])
            values.append(local_minima[index+1])
        else:
            values.append(None)
print(values) # the range of interst

crop_output_path = 'crop_image.png' 
crop_output_path2 = 'crop_image_lower.png' 
crop_and_merge(image_path, values, crop_output_path,0.5, crop_output_path2)

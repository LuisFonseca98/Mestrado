from captcha.image import ImageCaptcha
import random
import string
from PIL import ImageFilter
"""
Generates a random string, giving random numbers
"""
def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

"""
Generates an captcha image (jpg) with random width and height
"""


def generate_captcha(length, width, height, image_format='jpg'):
    image = ImageCaptcha(width=width, height=height)
    captcha_text = generate_random_string(length)

    if image_format == 'jpg':
        image_path = f'SavedImages/{captcha_text}.jpg'
    elif image_format == 'png':
        image_path = f'SavedImages/{captcha_text}.png'
    else:
        raise ValueError("Invalid image format. Supported formats are 'jpg' and 'png'.")

    data = image.generate_image(captcha_text)
    image.write(captcha_text, image_path)

    # Add Gaussian noise
    data = data.filter(ImageFilter.GaussianBlur(radius=0.4))
    return image_path


"""
Splits the text received in the captcha 
"""


def split_text(generate_image):
    split_name = generate_image.split('/')
    split_name[1] = split_name[1].replace('.jpg', '')
    print('split', split_name)
    return split_name[1]


if __name__ == "__main__":

    length_of_sequence = random.randint(1, 9)
    randomWidth = random.randint(100, 300)
    randomHeight = random.randint(100, 300)
    generate_random_string(length_of_sequence)

    ###############jpg files################
    generate_captcha(length_of_sequence, randomWidth, randomHeight)
    generate_captcha(length_of_sequence, randomWidth, randomHeight)
    generate_captcha(length_of_sequence, randomWidth, randomHeight)
    generate_captcha(length_of_sequence, randomWidth, randomHeight)
    generate_captcha(length_of_sequence, randomWidth, randomHeight)

    generate_captcha(length_of_sequence, randomWidth, randomHeight,'png')
    generate_captcha(length_of_sequence, randomWidth, randomHeight,'png')
    generate_captcha(length_of_sequence, randomWidth, randomHeight,'png')
    generate_captcha(length_of_sequence, randomWidth, randomHeight,'png')
    generate_captcha(length_of_sequence, randomWidth, randomHeight,'png')
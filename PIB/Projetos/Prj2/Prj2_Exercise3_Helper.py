from captcha.image import ImageCaptcha
import random
import string

"""
Generates a random string, giving random numbers
"""


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


"""
Generates an captcha image with random width and height
"""


def generate_captcha(length, width, height):
    image = ImageCaptcha(width=width, height=height)
    captcha_text = generate_random_string(length)
    image_path = f'SavedImages/{captcha_text}.png'
    data = image.generate_image(captcha_text)
    image.write(captcha_text, image_path)
    return image_path


"""
Splits the text received in the captcha 
"""


def split_text(generate_image):
    split_name = generate_image.split('/')
    split_name[1] = split_name[1].replace('.png', '')
    print(split_name)
    return split_name[1]

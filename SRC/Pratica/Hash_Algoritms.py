import hashlib
import base64
import binascii
from pyDes import des, triple_des, PAD_PKCS5

def convert_message_to_hash(message, hash_function):

    global hash_object

    if hash_function == 'md5':
        hash_object = hashlib.md5()
    elif hash_function == 'sha1':
        hash_object = hashlib.sha1()
    elif hash_function == 'sha256':
        hash_object = hashlib.sha256()

    hash_object.update(message.encode())
    result_hash = hash_object.hexdigest()

    print("Result Hash with " + hash_function + ": ", result_hash)


def convert_message_to_hash_with_base64(message, hash_function):
    global hash_object
    if hash_function == 'md5':
        hash_object = hashlib.md5(message.encode()).digest()
    elif hash_function == 'sha1':
        hash_object = hashlib.sha1(message.encode()).digest()
    elif hash_function == 'sha256':
        hash_object = hashlib.sha256(message.encode()).digest()
    result_hash_base64 = base64.b64encode(hash_object)

    print("Result Hash with " + hash_function + " using base64: ", result_hash_base64.decode())


def DES_encryption(message, des_key, isToDEncrypt):
    des_cipher = des(des_key, padmode=PAD_PKCS5)
    encrypted_data_des = des_cipher.encrypt(message.encode())
    if isToDEncrypt:
        encrypted_data_des_hex = binascii.hexlify(encrypted_data_des)
        print('DES Encrypted Message: ', encrypted_data_des_hex.decode())
        return encrypted_data_des_hex.decode()
    else:
        decrypted_data_des = des_cipher.decrypt(encrypted_data_des)
        print('DES Decrypted Message: ', decrypted_data_des.decode())
        return decrypted_data_des.decode()

def triple_DES_encryption(message, des_key, isToDEncrypt):
    triple_des_cipher = triple_des(des_key, padmode=PAD_PKCS5)
    encrypted_data_triple_des = triple_des_cipher.encrypt(message.encode())
    if isToDEncrypt:
        encrypted_data_triple_des_hex = binascii.hexlify(encrypted_data_triple_des)
        print('Triple DES Encrypted Message: ', encrypted_data_triple_des_hex.decode())
        return encrypted_data_triple_des
    else:
        decrypted_data_triple_des = triple_des_cipher.decrypt(encrypted_data_triple_des)
        print('Triple DES Decrypted Message: ', decrypted_data_triple_des.decode())
        return decrypted_data_triple_des


message = "Hello World!"
des_key = b"12345678"
triple_des_key = b"123456781234567812345678"

convert_message_to_hash(message, 'md5')
convert_message_to_hash(message, 'sha1')
convert_message_to_hash(message, 'sha256')

print("")
convert_message_to_hash_with_base64(message, 'md5')
convert_message_to_hash_with_base64(message, 'sha1')
convert_message_to_hash_with_base64(message, 'sha256')
print("")

DES_encryption(message,des_key, True)
DES_encryption(message, des_key, False)
print("")

triple_DES_encryption(message, triple_des_key, True)
triple_DES_encryption(message, triple_des_key, False)


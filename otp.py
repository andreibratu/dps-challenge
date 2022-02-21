import base64
import hashlib

import pyotp

secret = base64.b32encode(bytearray("thisisobviouslysecret", 'ascii')).decode('utf-8')

print(secret)

totp = pyotp.TOTP(secret, digest=hashlib.sha512, interval=120, digits=10)

print(totp.now())
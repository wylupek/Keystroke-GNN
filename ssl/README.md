### Generate SSL
Generate all necessary certificates using the following commands:
```
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -config san.cnf -extensions req_ext
openssl x509 -outform der -in cert.pem -out cert.crt
```
### Server
Put `key.pem` and `cert.pem` into `/ssl/`. \
Create `/.env` file with `SSL_PASSPHRASE=<passphrase>` in it.

### Client
Create a `res/raw` directory in your Android project, and place `cert.crt` there.
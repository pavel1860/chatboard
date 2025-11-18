# pip install pycryptodomex
import base64, json, hmac, hashlib
from Crypto.Cipher import AES

INFO = b"NextAuth.js Generated Encryption Key"

def b64u_decode(s: str) -> bytes:
    # base64url decode with safe padding
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode())

def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    if not salt:
        salt = b"\x00" * hashlib.sha256().digest_size
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    t = b""
    okm = b""
    counter = 0
    while len(okm) < length:
        counter += 1
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm += t
    return okm[:length]

def derive_key(nextauth_secret: str, *, salt: bytes = b"") -> bytes:
    # same HKDF params NextAuth uses
    return hkdf_sha256(nextauth_secret.encode(), salt, INFO, 32)

def decode_nextauth_session_token(token: str, nextauth_secret: str, salt: bytes = b"") -> dict:
    """
    Decrypt NextAuth JWE (dir/A256GCM) cookie: next-auth.session-token / __Secure-next-auth.session-token
    Returns dict payload (e.g., {'sub':..., 'email':..., 'exp':...})
    """
    parts = token.split(".")
    if len(parts) != 5:
        raise ValueError("Not a compact JWE (expected 5 parts).")

    protected_b64, enc_key_b64, iv_b64, ct_b64, tag_b64 = parts
    if enc_key_b64 not in ("", None):
        # For 'dir', the Encrypted Key part should be empty
        raise ValueError("Unexpected Encrypted Key for alg=dir")

    aad = protected_b64.encode()
    header = json.loads(b64u_decode(protected_b64).decode())

    if header.get("alg") != "dir" or header.get("enc") != "A256GCM":
        raise ValueError(f"Unsupported JWE header: {header}")

    iv = b64u_decode(iv_b64)
    ciphertext = b64u_decode(ct_b64)
    tag = b64u_decode(tag_b64)

    key = derive_key(nextauth_secret, salt=salt)

    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    cipher.update(aad)  # AAD is the protected header (base64url-encoded), per JWE spec
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    return json.loads(plaintext.decode())

# Example:
# payload = decode_nextauth_session_token(token, os.environ["NEXTAUTH_SECRET"])

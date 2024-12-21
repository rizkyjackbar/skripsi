import sys
import os

# Tambahkan path virtual environment ke sys.path
sys.path.insert(0, '/Users/macrizkyjackbar/Pribadi/Skripsi/stresslevel/venv/lib/python3.12/site-packages')

# Tambahkan folder aplikasi ke sys.path
sys.path.insert(1, '/Users/macrizkyjackbar/Pribadi/Skripsi/stresslevel')

# Import aplikasi Flask
from app import app as application

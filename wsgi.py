import sys
import os

# Menambahkan path virtual environment ke sys.path
sys.path.insert(0, '/Users/macrizkyjackbar/Pribadi/Skripsi/stresslevel/venv/lib/python3.12/site-packages')

# Menambahkan folder aplikasi ke sys.path
sys.path.insert(0, '/Users/macrizkyjackbar/Pribadi/Skripsi/stresslevel')

# Menambahkan aplikasi Flask ke dalam wsgi
from app import app as application

# 2. Chạy Visdom server ngầm
import subprocess
import time

visdom_proc = subprocess.Popen(["python3", "-m", "visdom.server", "-port", "8097"])
time.sleep(3)  # Chờ Visdom khởi động
# Nếu vẫn dùng localtunnel:
lt_proc = subprocess.Popen(['lt', '--port', '8097'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 4. Lấy URL từ stdout
import re

while True:
    line = lt_proc.stdout.readline()
    if 'your url is:' in line.lower():
        url = re.search(r'(https?://[^\s]+)', line).group(1)
        print("✅ Public Visdom URL:", url)
        break

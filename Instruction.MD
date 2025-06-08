# Hướng Dẫn Sử Dụng Máy Chủ Ubuntu Để Training Mô Hình

---

## 1. Tạo Máy Chủ Ubuntu

- Lưu lại thông tin IP, username, password

## 2. Kết Nối SSH Từ Windows

- Mở PowerShell trên Windows.
- Kết nối tới máy chủ bằng lệnh:

```bash
ssh username@192.x.x.x
```

## 3. Copy Dataset Từ Windows Lên Ubuntu

- Mở thêm một cửa sổ PowerShell mới trên Windows.
- Copy file dataset đã nén (ví dụ file.zip) lên Ubuntu:

```bash
scp -r "C:\yourroot\file.zip" username@192.x.x.x:~/Projects
```

---

## 4. Giải Nén Dataset Trên Ubuntu

- Quay lại cửa sổ SSH trên Ubuntu.
- Giải nén file đã copy:

```bash
unzip ~/Projects/file.zip -d ~/Projects/
```

---

## 5. Clone Source Code Từ GitHub

- Trong cửa sổ SSH Ubuntu, chạy:

```bash
git clone https://github.com/longnguyenha050/CS331.git
```

---

## 6. Cài Đặt Thư Viện Và Thiết Lập Môi Trường

- Vào thư mục dự án:

```bash
cd CS331
```

- Đọc và làm theo hướng dẫn trong `README.md` để cài đặt thư viện và môi trường.

---

## 7. Bắt Đầu Training

- Chạy lệnh training theo hướng dẫn, ví dụ:

```bash
python3 train.py --no_flip --no_rotation --no_augment --img_file ./Projects --niter 450000 --niter_decay 200000 --lr 1e-4
```

---

## 8. Copy Thư Mục Checkpoint Từ Ubuntu Về Windows

- Mở PowerShell khác trên Windows.
- Copy thư mục checkpoint từ Ubuntu về Windows, ví dụ:

```bash
scp -r username@192.x.x.x:/path/to/remote/checkpoint/folder "C:\path\to\local\folder"
```

---

## Lưu Ý

- Thay thế các thông số như `username`, `192.x.x.x`, đường dẫn cho phù hợp với máy bạn.
- Đảm bảo các lệnh `ssh` và `scp` hoạt động bình thường.
- Nếu có firewall hoặc mạng giới hạn, cần mở port hoặc cấu hình phù hợp.

---

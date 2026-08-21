# Deploy lên AWS EC2

Repo dùng bucket `income-cicd-394877251429` và region mặc định
`ap-southeast-1`. EC2 nên dùng Ubuntu 22.04/24.04, loại `t2.micro` hoặc
`t3.micro`.

## 1. Quyền và mạng

- Gắn IAM role có policy trong `ec2-s3-read-policy.json` vào EC2. VM chỉ cần
  đọc model production, không cần lưu AWS access key trên máy.
- Security Group inbound: SSH/TCP 22 chỉ từ IP của bạn; Custom TCP 8080 từ
  IP cần kiểm thử. Không mở SSH `0.0.0.0/0`.
- Đảm bảo instance có public IPv4 hoặc Elastic IP.

## 2. Khởi tạo VM một lần

SSH vào VM, clone repo rồi chạy:

```bash
git clone https://github.com/PhamDuong2705/K4-Track2-Day21-2A202601404-PhamTungDuong.git
cd K4-Track2-Day21-2A202601404-PhamTungDuong
bash deploy/bootstrap-ec2.sh income-cicd-394877251429
```

Thêm public key tương ứng với `SERVER_SSH_KEY` vào `~/.ssh/authorized_keys`.
Script tạo virtualenv và service `income-api`; CI sẽ cập nhật code, promote
model đã qua quality gate lên S3 rồi restart service.

## 3. GitHub Actions secrets

Tạo đúng 5 repository secrets:

- `STORAGE_CREDENTIALS`: JSON một dòng, ví dụ
  `{"aws_access_key_id":"...","aws_secret_access_key":"...","region":"ap-southeast-1"}`.
- `ARTIFACT_BUCKET`: `income-cicd-394877251429`.
- `SERVER_HOST`: public IPv4/Elastic IP của EC2.
- `SERVER_USER`: thường là `ubuntu`.
- `SERVER_SSH_KEY`: toàn bộ private OpenSSH deploy key.

IAM user của GitHub Actions cần quyền đọc `dvc/*` và ghi
`artifacts/current/*` trong bucket. Không commit bất kỳ secret/private key nào.

## 4. Kiểm tra

Sau khi workflow `Income Model CI/CD` có đủ bốn jobs màu xanh:

```bash
curl http://<EC2_PUBLIC_IP>:8080/healthz
curl -X POST http://<EC2_PUBLIC_IP>:8080/score \
  -H 'Content-Type: application/json' \
  -d '{"features":[28,2,14,2,11,0,1,0,0,45]}'
```

Nếu health check lỗi, trên VM chạy:

```bash
sudo systemctl status income-api
sudo journalctl -u income-api --no-pager -n 100
```

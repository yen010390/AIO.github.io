---
title: "[Test]Module 1: Ứng dụng RAG trong việc hỏi đáp tài liệu bài học AIO"
excerpt: "Dự án này xây dựng hệ thống hỏi đáp thông minh dùng kiến trúc RAG, giúp người học khóa AI tại AI Việt Nam (AIO) khai thác hiệu quả nội dung tài liệu học tập."
collection: project
author: "Nguyễn Tuấn Anh - Đoàn Tấn Hưng - Hồ Thị Ngọc Huyền - Trần Thị Mỹ Tú - Đặng Thị Hoàng Yến"
tags:
- RAG
- LLM
- Chatbot
---

**Tác giả**: Nguyễn Tuấn Anh - Đoàn Tấn Hưng - Hồ Thị Ngọc Huyền - Trần Thị Mỹ Tú - Đặng Thị Hoàng Yến

<details>
<summary><strong>📁 Cấu trúc source code  (click để xem)</strong></summary>

Source code và tài liệu có thể được tìm thấy tại:
- [GitHub Repository](https://github.com/aio25-mix002/m01-p0102)
- [Jupyter Notebook hướng dẫn](https://github.com/aio25-mix002/m01-p0102/blob/main/runbook_m01p0102.ipynb)

<br>

```python
📦 RAG_AIO_Chatbot
├── assets/                   # Tài sản tĩnh (logo, favicon...)
│   └── logo.png              # Logo của ứng dụng
├── examples/                 # Dữ liệu mẫu để test
│   └── YOLOv10_Tutorials.pdf # File PDF mẫu
├── logs/                     # Thư mục lưu log
├── prompt_templates/         # Các template prompt cho RAG
├── utils/                    # Các tiện ích hỗ trợ
│   ├── logging_utils.py      # Utility logging
│   └── prompt_utils.py       # Utility quản lý prompt
├── .vscode/                  # Cấu hình Visual Studio Code
│   └── launch.json           # Debug configuration
├── .env                      # Biến môi trường production
├── .env.example              # Template biến môi trường
├── .env.local                # Biến môi trường local
├── rag_chatbot.py            # File chính - Streamlit RAG chatbot
├── runbook_m01p0102.ipynb    # Jupyter notebook hướng dẫn
├── requirements.txt          # Dependencies chính
├── requirements-torch.txt    # Dependencies PyTorch
├── .gitignore                # Git ignore rules
└── README.md                 # Tài liệu hướng dẫn
```
</details>

<details>
<summary><strong>📁 Mục lục báo cáo (click để xem)</strong></summary>
<br>

- [Tóm tắt](#tóm-tắt)
- [1. Giới thiệu 🗂](#1-giới-thiệu-)
- [2. Phương pháp luận 📚](#2-phương-pháp-luận-)
  - [2.1. Quy trình Lập chỉ mục dữ liệu (Indexing)](#21-quy-trình-lập-chỉ-mục-dữ-liệu-indexing)
  - [2.2. Quy trình Truy vấn và Tạo sinh (Retrieval \& Generation)](#22-quy-trình-truy-vấn-và-tạo-sinh-retrieval--generation)
- [3. Thực hiện ⚙](#3-thực-hiện-)
- [4. Kết quả 📈](#4-kết-quả-)
- [5. Mở rộng nâng cao 🖥](#5-mở-rộng-nâng-cao-)
  - [5.1 Tiêu chí cải tiến](#51-tiêu-chí-cải-tiến)
  - [5.2 Code nâng cao](#52-code-nâng-cao)
    - [5.2.1 Nâng cấp cốt lỗi: Ghi nhớ lịch sử hội thoại (Conversation memory)](#521-nâng-cấp-cốt-lỗi-ghi-nhớ-lịch-sử-hội-thoại-conversation-memory)
    - [5.2.2 Quản lý Vector DB nâng cao](#522-quản-lý-vector-db-nâng-cao)
    - [5.2.3. Gỡ lỗi (Debugging) với Logger](#523-gỡ-lỗi-debugging-với-logger)
    - [5.2.4. Xử lý và truy vấn từ nhiều file tài liệu](#524-xử-lý-và-truy-vấn-từ-nhiều-file-tài-liệu)
  - [5.3 Kết quả mở rộng 📍](#53-kết-quả-mở-rộng-)
    - [5.3.1 Hỗ trợ ghi nhớ](#531-hỗ-trợ-ghi-nhớ)
    - [5.3.2 Xử dụng tập tài liệu khác ứng dụng trong y khoa](#532-xử-dụng-tập-tài-liệu-khác-ứng-dụng-trong-y-khoa)
    - [5.3.3 Hỗ trợ làm việc với nhiều tài liệu khác nhau](#533-hỗ-trợ-làm-việc-với-nhiều-tài-liệu-khác-nhau)
- [6. Kết luận 📌](#6-kết-luận-)

</details>

<br>
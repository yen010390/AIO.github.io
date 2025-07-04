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
<summary><strong>📁 Cấu trúc source code (click để xem)</strong></summary>

</details>


Source code và tài liệu có thể được tìm thấy tại:
- [GitHub Repository](https://github.com/aio25-mix002/m01-p0102)
- [Jupyter Notebook hướng dẫn](https://github.com/aio25-mix002/m01-p0102/blob/main/runbook_m01p0102.ipynb)

<br>

```
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


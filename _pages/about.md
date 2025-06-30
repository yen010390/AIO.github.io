---
permalink: /
title: "Introduction"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<style>
/* Container for the team members */
.profile-container {
  display: flex; /* Use flexbox for alignment */
  gap: 1.5rem; /* Space between profile cards */
  justify-content: center; /* Center the cards horizontally */
  align-items: stretch; /* Make all cards have the same height */
  padding: 1rem 0; /* Add some vertical padding */
}

/* Individual profile card */
.profile {
  flex: 1; /* This is the key change: allows each card to grow and share space equally */
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  padding: 1.5rem 1rem;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
  text-align: center; /* Center the content inside the card */
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.profile:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 12px rgba(0,0,0,0.1);
}

/* Profile image styling */
.profile img {
  max-width: 120px; /* Slightly smaller image */
  height: 120px;
  object-fit: cover;
  border-radius: 50%; /* Circular image */
  margin: 0 auto 1rem auto; /* Center image horizontally */
  border: 3px solid #f0f0f0;
}

.profile h2 {
    font-size: 1.1em;
    margin-bottom: 0.5rem;
}

.profile p {
    font-size: 0.9em;
    word-break: break-word; /* Break long links if necessary */
}
</style>

<p><br>
  👋 Xin chào, chúng tôi đang trong quá trình xử lý... học AI!<br>
  👭 Nhóm của chúng tôi có 5 thành viên.<br>
  Hãy xem trang web của chúng tôi.</p>
<hr />

<!-- The container for the 5 member profiles -->
<div class="profile-container">

  <!-- Member 1 -->
  <div class="profile">
    <img src="/AIO.github.io/images/profile/profile-Anh.jpg" alt="Nguyễn Tuấn Anh" onerror="this.onerror=null;this.src='https://placehold.co/120x120/EFEFEF/333333?text=Anh';">
    <h2>Nguyễn Tuấn Anh</h2>
    <p><strong>👋 Github:</strong> <a href="https://github.com/yen010390" target="_blank" rel="noopener noreferrer">yen010390</a></p>
  </div>

  <!-- Member 2 -->
  <div class="profile">
    <img src="/AIO.github.io/images/profile/profile-Hung.jpg" alt="Hưng Đoàn" onerror="this.onerror=null;this.src='https://placehold.co/120x120/EFEFEF/333333?text=Hưng';">
    <h2>Hưng Đoàn</h2>
    <p><strong>👋 Github:</strong> <a href="https://github.com/hdaio25" target="_blank" rel="noopener noreferrer">hdaio25</a></p>
  </div>

  <!-- Member 3 -->
  <div class="profile">
    <img src="/AIO.github.io/images/profile/profile-Ngoc.jpg" alt="Ngọc Huyền" onerror="this.onerror=null;this.src='https://placehold.co/120x120/EFEFEF/333333?text=Ngọc';">
    <h2>Ngọc Huyền</h2>
    <p><strong>👋 Github:</strong> <a href="https://github.com/ngochuyen2723" target="_blank" rel="noopener noreferrer">ngochuyen2723</a></p>
  </div>

  <!-- Member 4 -->
  <div class="profile">
    <img src="/AIO.github.io/images/profile/profile-Tu.jpg" alt="Trần Thị Mỹ Tú" onerror="this.onerror=null;this.src='https://placehold.co/120x120/EFEFEF/333333?text=Tú';">
    <h2>Trần Thị Mỹ Tú</h2>
    <p><strong>👋 Github:</strong> <a href="https://github.com/daria-tran" target="_blank" rel="noopener noreferrer">daria-tran</a></p>
  </div>
  
  <!-- Member 5 -->
  <div class="profile">
    <img src="/AIO.github.io/images/profile/profile-Yen.jpg" alt="Đặng Thị Hoàng Yến" onerror="this.onerror=null;this.src='https://placehold.co/120x120/EFEFEF/333333?text=Yến';">
    <h2>Đặng Thị Hoàng Yến</h2>
    <p><strong>👋 Github:</strong> <a href="https://github.com/yen010390" target="_blank" rel="noopener noreferrer">yen010390</a></p>
  </div>

</div>

<hr />

<h2>🎯 Giới thiệu về trang web của chúng tôi</h2>
<p>Trang web của chúng tôi được tạo ra để ghi lại hành trình học Trí tuệ nhân tạo của chúng tôi, từ kiến thức cơ bản đến các dự án ứng dụng. Bạn có thể khám phá các phần khác nhau bên dưới:</p>
<ul>
  <li><a href="/AIO.github.io/posts/">📝 Blog Posts</a> – Thông tin chi tiết, bài học và suy ngẫm về chủ đề AI.</li>
  <li><a href="/AIO.github.io/project/">🛠 Projects</a> – Các dự án thực hành giới thiệu những gì chúng tôi đã xây dựng.</li>
</ul>

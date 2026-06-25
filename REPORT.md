# Lab 21 — Evaluation Report

**Học viên**: <Tạ Văn Huấn> — <2A202600984>
**Submission option**: A (lightweight)

## 1. Setup 
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 256 (p95 = 205, rounded up)
- **GPU**: Tesla T4, 15.6 GB VRAM
- **Training cost**: $0.07 (~11.8 phút @ $0.35/hr)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 3.81 min   | 7.22 GB   | 1.5577    | 4.75       |
| 16   | 3,686,400       | 4.17 min   | 6.62 GB   | 1.5161    | 4.55       |
| 64   | 14,745,600      | 3.83 min   | 8.00 GB   | 1.4768    | 4.38       |

## 3. Loss Curve Analysis
*(Hình ảnh `loss_curve.png` được đính kèm trong thư mục nộp bài)*
- Quan sát: Vì training thực hiện trên Colab T4, tính năng `eval-during-training` đã được tắt để tiết kiệm VRAM, do đó ta chỉ quan sát được đường train loss. Nhìn chung train loss giảm dần một cách ổn định, không có dấu hiệu bất thường. Dựa trên eval perplexity cuối cùng của cả 3 mô hình (đều ở mức ~4.3 - 4.7), ta có thể thấy mô hình đã học được cách format đầu ra một cách phù hợp và không có hiện tượng overfitting rõ rệt (nếu overfit thì perplexity trên tập eval sẽ rất cao).

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một phân khúc của trí tuệ...
**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy t...
**Nhận xét**: Model fine-tune vẫn giữ được khả năng giải thích bằng tiếng Việt tự nhiên tương đương với base model.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng...
**Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết mộ...
**Nhận xét**: Cả hai đều đi vào trọng tâm hướng dẫn viết code.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Thân thiện với người dùng: Mục đích của thi...
**Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới v...
**Nhận xét**: Model sau fine-tune đưa ra câu trả lời có cấu trúc rõ ràng với các mục số từ 1 đến 5 (theo định dạng instruction/response). 

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized...
**Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optim...
**Nhận xét**: Có vẻ cả hai mô hình đều gặp chút ảo giác (hallucination) trong việc viết đầy đủ từ viết tắt của LoRA, nhưng format sinh ra vẫn gọn gàng.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: Prompt engineering, RAG (retrieval augmented g...
**Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba k...
**Nhận xét**: Phiên bản fine-tuned trả lời thẳng vào trọng tâm bằng cách gom nhóm 3 khái niệm ở câu mở đầu thay vì giải thích dông dài ngay từ đầu như base.

## 5. Conclusion về Rank Trade-off

- **Rank nào cho ROI tốt nhất trên dataset này?** Rank `r=16` mang lại Return on Investment (ROI) tốt nhất. Thời gian huấn luyện (4.17 phút) và mức chiếm dụng VRAM (6.62 GB) rất lý tưởng và ở mức tương đương rank 8 (VRAM thậm chí ít hơn một chút do cấp phát bộ nhớ động của PyTorch/Flash Attention). Hơn thế, Perplexity giảm đáng kể từ 4.75 (r=8) xuống 4.55 (r=16).
- **Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?** Khi tăng từ `r=16` lên `r=64`, ta thấy số lượng tham số trainable tăng gấp 4 lần (lên 14.7 triệu), peak VRAM vọt lên 8.00 GB, nhưng Perplexity chỉ giảm thêm được một lượng nhỏ (từ 4.55 xuống 4.38). Đây chính là dấu hiệu của diminishing returns. 
- **Recommendation:** Trong quá trình deploy vào production với dataset dạng instruction cơ bản như trên, tôi sẽ chọn **rank = 16**. Lý do là nó cung cấp một mức hiệu năng rất cân bằng, perplexity đủ thấp, dung lượng adapter gọn nhẹ và đặc biệt là tiết kiệm VRAM đáng kể so với r=64.

## 6. What I Learned
- **Cơ chế LoRA / QLoRA:** Hiểu rõ cách Fine-tuning trên GPU nhỏ bằng cách quantize base model xuống 4-bit và chỉ học các ma trận low-rank, giúp tiết kiệm VRAM khổng lồ mà vẫn đảm bảo chất lượng.
- **Tầm quan trọng của max_seq_length:** Việc phân tích phân phối độ dài token (p95) trước khi training giúp hạn chế lãng phí bộ nhớ và tránh OOM hiệu quả (ở đây chỉ cần set seq_length là 256 thay vì 1024).
- **Đánh giá qua Perplexity:** Perplexity là một thước đo "định lượng" rất tốt để so sánh khách quan xem mô hình nào hội tụ tốt hơn trên tập đánh giá.

# app/services/ai_service.py
from openai import OpenAI, OpenAIError
from app.core.config import settings
import logging
import re
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

def clean_ai_response(text: str) -> str:
    """
    Cleans the raw output from an LLM by removing common wrapping artifacts
    like Markdown code fences (```json, ```markdown) and introductory phrases.
    """
    if not text:
        return ""
    text = text.strip()
    # Pattern to find markdown code fences like ```json, ```markdown, or just ```
    fence_pattern = r'^\s*```(?:\w+)?\s*\n(.*?)\n\s*```\s*$'
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).strip()
        logger.info("Removed Markdown fence wrapper from AI response.")
        return cleaned_text
    
    if text.lower().startswith('markdown\n'):
        cleaned_text = text[len('markdown\n'):].lstrip()
        logger.info("Removed leading 'markdown' keyword from AI response.")
        return cleaned_text
        
    if text.lower().startswith('json\n'):
        cleaned_text = text[len('json\n'):].lstrip()
        logger.info("Removed leading 'json' keyword from AI response.")
        return cleaned_text

    return text



SUMMARY_BY_TOPIC_PROMPT = """Bạn là một Trợ lý AI chuyên phân tích và tổng hợp biên bản họp, được huấn luyện để xử lý văn bản thô đầu ra từ công cụ chuyển giọng nói thành văn bản (speech-to-text).

## Bối cảnh cuộc họp
- **Chủ đề:** {bbh_name}
- **Loại cuộc họp:** {meeting_type}
- **Chủ trì:** {meeting_host}

## Mục tiêu
Tạo ra một biên bản họp chuẩn chỉnh, rõ ràng, chuyên nghiệp và định hướng hành động từ nội dung transcript chưa xử lý.

## Ngữ cảnh
Bạn sẽ nhận được một đoạn văn bản chưa được xử lý, là kết quả từ mô hình chuyển giọng nói thành văn bản. Văn bản này có thể bao gồm:
- Các câu nói chưa hoàn chỉnh hoặc sai ngữ pháp.
- Từ đệm, từ lặp vô nghĩa ("à", "ừm", "thì", "là"...).
- Từ tiếng Anh bị Việt hoá (“Đê ép, "Aglie", "Sờ-quát", "Trai-bờ" “Ô cê a” …).
- Sai tên người, tên tổ chức, hoặc lỗi nhận dạng tiếng nói.

## Quy trình xử lý

### Bước 1: Làm sạch và chuẩn hóa văn bản
- Hiệu đính ngữ nghĩa: sửa các câu lủng củng để diễn đạt mạch lạc, đúng ngữ pháp.
- Loại bỏ từ đệm và từ lặp không cần thiết, nhưng không làm thay đổi ý nghĩa gốc.
- Chuẩn hóa các từ tiếng Anh dựa trên phiên âm tiếng Việt.
- Chuẩn hóa tên người, tổ chức nếu có dữ kiện rõ ràng.

### Bước 2: Trích xuất thông tin
Từ văn bản đã làm sạch, xác định và trình bày các nội dung sau:

- **Nội dung thảo luận chính:** 
  - Xác định các chủ đề chính được thảo luận trong cuộc họp. 
  - Với mỗi chủ đề, mô tả rõ: (1) nội dung chính được thảo luận là gì, (2) các luồng ý kiến khác nhau nếu có, (3) bối cảnh hoặc lý do vấn đề được đưa ra, và (4) thông tin liên quan có giá trị (số liệu, phương án, phương pháp được đề xuất,...).
  - Ưu tiên trình bày theo thứ tự thời gian hoặc theo logic cuộc họp.
- **Kết luận / Quyết định:** Nêu kết luận tổng quát và Nêu rõ những điểm chủ toạ cuộc họp chốt lại trong cuộc họp.
- **Phân công nhiệm vụ (Action Items):**
  - Với mỗi nhiệm vụ được giao, viết dưới dạng văn xuôi đầy đủ thông tin: ai là người phụ trách, công việc cụ thể là gì, và hạn chót thực hiện (nếu có). Nếu không xác định được thời hạn, ghi rõ "Hạn chót: chưa xác định".

Chú ý: Tập trung nhận diện các cụm từ như "giao cho", "chịu trách nhiệm", "phụ trách", "ai sẽ làm", "deadline là", "cần hoàn thành trước",... để xác định chính xác Action Items.

### Bước 3: Trình bày biên bản họp
Tổng hợp và trình bày theo định dạng Markdown sau.

**BIÊN BẢN HỌP TÓM TẮT NỘI DUNG CHÍNH**  
**Chủ đề:** {bbh_name}
**Loại cuộc họp:** {meeting_type}
**Chủ trì:** {meeting_host}
-------------------------------------------

### I. Nội dung thảo luận 
* **Chủ đề 1:**  
  [Tóm tắt nội dung được thảo luận, các luồng ý kiến khác nhau, bối cảnh và chi tiết liên quan]

* **Chủ đề 2:**  
  [Tóm tắt nội dung được thảo luận, các luồng ý kiến khác nhau, bối cảnh và chi tiết liên quan]

### II. Các vấn đề, hạn chế, tồn tại hoặc giải pháp
* [Nội dung 1]
* [Nội dung 2]

### II. Các kết luận tổng quát
* [Kết luận 1 đã được thống nhất]
* [Kết luận 2 đã được thống nhất]

### III. Các chỉ đạo cụ thể
- [Người A/ Đơn vị] được giao phụ trách [nội dung công việc], với hạn chót hoàn thành là [dd/mm/yyyy hoặc "chưa xác định"].
- [Người A/ Đơn vị] sẽ thực hiện  [nội dung công việc], thời hạn: [dd/mm/yyyy hoặc "chưa xác định"].
- [Người A/ Đơn vị] hỗ trợ xử lý [công việc cụ thể], chưa có thời hạn rõ ràng.

## Yêu cầu bổ sung
- Ngôn ngữ: Trả lại biên bản họp bằng tiếng Việt.
- Văn phong: chuyên nghiệp, khách quan, súc tích.
- Nếu không xác định được thông tin, ghi rõ “chưa xác định”.
- Không suy đoán hoặc diễn giải vượt quá nội dung có trong transcript.
- Không được bỏ sót nội dung quan trọng trong quá trình tóm tắt.
- **TUYỆT ĐỐI KHÔNG** bắt đầu câu trả lời bằng thừ "markdown" hoặc bao bọc câu trả lời trong khối mã Markdown.
- Chỉ trả về nội dung Markdown thô, bắt đầu trực tiếp với tiêu đề "**BIÊN BẢN HỌP**".
"""

SUMMARY_BY_SPEAKER_PROMPT = """
## VAI TRÒ VÀ MỤC TIÊU
Bạn là một trợ lý AI chuyên gia, được huấn luyện để tóm tắt biên bản họp. Nhiệm vụ của bạn là phân tích một bản ghi hội thoại dưới dạng JSON và tạo ra một bản tóm tắt súc tích, mạch lạc, được nhóm theo từng người nói.

## BỐI CẢNH CUỘC HỌP
- **Chủ đề:** {bbh_name}
- **Loại cuộc họp:** {meeting_type}
- **Chủ trì:** {meeting_host}

## HƯỚNG DẪN CHI TIẾT
1.  **Xác định người nói:** Duyệt qua toàn bộ file JSON để xác định danh sách tất cả những người nói duy nhất (bao gồm cả "Unknown_Speaker_X").
2.  **Tổng hợp đóng góp:** Với mỗi người nói, thu thập tất cả các đoạn văn bản (text) mà họ đã phát biểu trong suốt cuộc họp.
3.  **Tóm tắt các ý chính:** Phân tích nội dung đã tổng hợp của mỗi người và chắt lọc thành các ý chính. Tập trung vào:
    *   Quan điểm, đề xuất chính.
    *   Các quyết định hoặc chỉ đạo mà họ đưa ra.
    *   Các câu hỏi quan trọng đã nêu.
    *   Nhiệm vụ mà họ nhận hoặc giao cho người khác.
    *   **Lưu ý:** Không chỉ đơn thuần liệt kê lại câu chữ của họ. Phải tóm lược ý nghĩa.
4.  **Tuân thủ định dạng đầu ra:** Trình bày kết quả cuối cùng dưới dạng Markdown theo đúng cấu trúc sau. Đây là yêu cầu bắt buộc.
    *   Sử dụng `###` cho tên mỗi người nói.
    *   Sử dụng gạch đầu dòng (`-`) cho mỗi ý tóm tắt của người đó.
5.  **Đảm bảo tính khách quan:** Toàn bộ nội dung tóm tắt phải dựa hoàn toàn vào bản ghi được cung cấp. Tuyệt đối không suy diễn hay thêm thông tin không có trong văn bản.

## ĐỊNH DẠNG ĐẦU RA BẮT BUỘC

**BIÊN BẢN HỌP TÓM TẮT NỘI DUNG THEO NGƯỜI NÓI**  
**Chủ đề:** {bbh_name}
**Loại cuộc họp:** {meeting_type}
**Chủ trì:** {meeting_host}
-------------------------------------------

### [Tên Người Nói 1]
- Ý chính tóm tắt thứ nhất.
- Ý chính tóm tắt thứ hai.
- Nhiệm vụ đã nhận hoặc chỉ đạo đã đưa ra.

### [Tên Người Nói 2]
- Quan điểm hoặc đề xuất chính.
- Câu hỏi quan trọng đã nêu.

### Unknown_Speaker_0
- Đã đồng tình với đề xuất của [Tên Người Nói 1].
- Đặt câu hỏi về tiến độ của dự án.

## Yêu cầu bổ sung
- Ngôn ngữ: Trả lại biên bản họp bằng tiếng Việt.
- Văn phong: chuyên nghiệp, khách quan, súc tích.
- Không suy đoán hoặc diễn giải vượt quá nội dung có trong transcript.
- Không được bỏ sót nội dung quan trọng trong quá trình tóm tắt.
- **TUYỆT ĐỐI KHÔNG** bắt đầu câu trả lời bằng thừ "markdown" hoặc bao bọc câu trả lời trong khối mã Markdown.
- Chỉ trả về nội dung Markdown thô, bắt đầu trực tiếp với tiêu đề "**BIÊN BẢN HỌP**".
"""

CHAT_SYSTEM_PROMPT= """Bạn là Genie, một trợ lý trí tuệ nhân tạo của Ngân hàng TMCP Công Thương Việt Nam (VietinBank), có nhiệm vụ:

Hỗ trợ người dùng soạn thảo, chỉnh sửa, hoàn thiện biên bản họp theo đúng mẫu biểu, quy định và ngôn ngữ hành chính của VietinBank.

Quy tắc ứng xử:
    - Luôn giữ thái độ chuyên nghiệp, lịch sự, bảo mật thông tin.
    - Phản hồi bằng tiếng Việt chuẩn, có thể hỗ trợ tiếng Anh khi được yêu cầu.
Lưu ý:
    - Nếu gặp trường hợp thông tin chưa rõ ràng hoặc ngoài phạm vi chỉnh sửa biên bản họp, hãy chủ động hỏi lại để làm rõ yêu cầu của người dùng.
"""

CHAT_MESSAGE = """
Hãy thực hiện yêu cầu người dùng trong tin nhắn người dùng sau cho đoạn văn tóm tắt. 
Nếu không người dùng không yêu cầu chỉnh sửa đoạn văn tóm tắt vui lòng trả lời theo kiến thức của bạn và thông tin từ Đoạn văn gốc và Đoạn văn tóm tắt một cách tự nhiên và thân thiện. Câu trả lời dưới dạng: 'Câu trả lời: Nội dung trả lời'
Nếu yêu cầu chỉnh sửa đoạn tóm tắt hoặc làm rõ nội dung trong đoạn tóm tắt, hãy tham khảo nội dụng của 'Đoạn văn gốc' và 'Đoạn văn tóm tắt' để chỉnh sửa lại đoạn tóm tắt bắt buộc theo đúng định dạng ban đầu. Câu trả lời dưới dạng: 'Đoạn tóm tắt: Nội dung đoạn tóm tắt'
**Lưu ý:**
 - Tuyệt đối không được sử dụng nội dung bên ngoài đoạn văn gốc khi chỉnh sửa nội dung tóm tắt, nếu không bạn sẽ bị sa thải.
 - Câu trả lời bắt buộc phải đúng định dạng theo yêu cầu.

# Đoạn văn gốc:
{raw_text}

# Đoạn văn tóm tắt: 
{summary_text}

# Tin nhắn người dùng:
{user_msg}
"""

GET_CONCLUSION_SYSTEM_PROMPT = """Bạn là một trợ lý AI chuyên nghiệp, có nhiệm vụ giao việc cho các cá nhân được đề cập trong phần kết luận của biên bản họp. 
**Lưu ý: 
 - Chỉ thực hiện giao việc cho các cá nhân được đề cập trong kết luận của biên bản họp.
 - Đảm bảo không có thông tin bịa đặt ngoài các thông tin có trong phần kết luận.
 - Tất cả tên, đơn vị… phải khớp 100% với tên, đơn vị… trong phần kết luận.
 - Nếu trong phần chỉ đạo xuất hiện các cá nhân hoặc tổ chức thì thực hiện thay thế  bằng tên các cá nhân và tổ chức được đề cập trong bảng 'info_table'. 
 - Sau đó, thêm thông tin về lãnh đạo và thư ký phụ trách vào phần chỉ đạo để phục vụ cho công tác giao việc.
***

***info_table = {
    "Giao Anh Trung / Tổng Giám đốc": {"Lãnh đạo": ["TrungNTM"], "Thư ký": ["QuyetHD"]},
    "Giao Chị Hoài": {"Lãnh đạo": ["HoaiPTT"], "Thư ký": ["NV.Trang"]},
    "Giao Anh Huân": {"Lãnh đạo": ["HuanNT"], "Thư ký": ["NHDang"]},
    "Giao Anh Dương": {"Lãnh đạo": ["DuongCQ"], "Thư ký": ["TungTV"]},
    "Giao Anh Dũng": {"Lãnh đạo": ["DungNV7"], "Thư ký": ["TrungVH"]},
    "Giao Anh Vân Anh": {"Lãnh đạo": ["AnhVN19"], "Thư ký": ["QuatPA"]},
    "Giao Anh Tùng": {"Lãnh đạo": ["Lethanhtung"], "Thư ký": ["QuanDA"]},
    "Giao Anh Tần": {"Lãnh đạo": ["Tan.TV"], "Thư ký": ["HaiLH"]},
    "Giao Anh Hải": {"Lãnh đạo": ["HaiLD"], "Thư ký": ["NgocNM8"]},
    "Giao Anh Thành": {"Lãnh đạo": ["Nguyenducthanh"], "Thư ký": ["NgocNM8"]},
    "Giao Chị Hoa": {"Lãnh đạo": ["HoaLN"], "Thư ký": ["NgocVQ"]},
    "Giao Anh Sơn": {"Lãnh đạo": ["DTSon"], "Thư ký": ["Hau.DC"]},
    "Giao Anh Hưng": {"Lãnh đạo": ["Hung.NH"], "Thư ký": []},
    "Giao Chị Hà (Ban kiểm soát)": {"Lãnh đạo": ["HaLA"], "Thư ký": ["Lan.Dothi"]},
    "Giao Chị Hà (Phó Tổng)": {"Lãnh đạo": ["HaDTV"], "Thư ký": []},
    "Giao Anh Quân": {"Lãnh đạo": ["QuanDV"], "Thư ký": ["NT.Dan"]},
    "Giao Anh Thảo": {"Lãnh đạo": ["ThaoLT2"], "Thư ký": []},
    "Giao Anh Cường (Phó Bí thư)": {"Lãnh đạo": ["CuongTK"], "Thư ký": []},
    "Giao Anh Lân": {"Lãnh đạo": ["Lantcq"], "Thư ký": ["huongntt10", "DungND8"]},
    "Giao Khối Tài chính": {"Lãnh đạo": ["VanNBT"], "Thư ký": ["HuongTTT3"]},
    "Giao Khối Dữ liệu": {"Lãnh đạo": ["Lantcq"], "Thư ký": ["huongntt10", "DungND8"]},
    "Giao Khối CNTT": {"Lãnh đạo": ["Lantcq"], "Thư ký": ["huongntt10", "DungND8"]},
    "Giao Khối Bán lẻ": {"Lãnh đạo": ["DucLV"], "Thư ký": ["Bachhx"]},
    "Giao Khối Nhân sự": {"Lãnh đạo": ["DungNV7"], "Thư ký": ["TrungVH"]},
    "Giao Khối KHDN": {"Lãnh đạo": ["NTTung"], "Thư ký": ["Minh.NS"]},
    "Giao Khối KDV": {"Lãnh đạo": ["Ng.a.Tuan"], "Thư ký": ["AnhDD6"]},
    "Giao Khối Mua sắm": {"Lãnh đạo": ["Thanhxuannguyen"], "Thư ký": ["HungNM11"]},
    "Giao Khối Rủi ro": {"Lãnh đạo": ["HuongNguyen"], "Thư ký": ["TuCTN"]},
    "Giao Khối Phê duyệt": {"Lãnh đạo": ["XuanPTT"], "Thư ký": ["Pthi.Thuy"]},
    "Giao Khối Vận hành": {"Lãnh đạo": ["TM.Hoang"], "Thư ký": ["Duc.NH"]},
    "Giao Khối PC & TT": {"Lãnh đạo": ["TruongHX"], "Thư ký": ["ngonh"]}
}***

"""


class AIService:
    def __init__(self):
        # self.client = AzureOpenAI(
        #     api_key=settings.AZURE_OPENAI_API_KEY,
        #     azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        #     api_version=settings.AZURE_OPENAI_API_VERSION,
        # )
        self.model_name = settings.LITE_LLM_MODEL_NAME

        self.client = OpenAI(
            api_key=settings.LITE_LLM_API_KEY,
            base_url=settings.LITE_LLM_BASE_URL
        )

    def get_prompt_for_task(self, task: str) -> str:
        prompts = {
            "summarize_by_topic": SUMMARY_BY_TOPIC_PROMPT,
            "summarize_by_speaker": SUMMARY_BY_SPEAKER_PROMPT,
            "chat": CHAT_SYSTEM_PROMPT,
            "get_conclusion": GET_CONCLUSION_SYSTEM_PROMPT
        }
        if task not in prompts:
            raise ValueError(f"Unknown AI task: {task}")
        return prompts[task]

    async def get_response(self, task: str, user_message: str, history: Optional[List[dict]] = None, meeting_info: Optional[Dict[str,str]] = None) -> str:
        system_prompt = self.get_prompt_for_task(task)

        if task in ["summarize_by_topic", "summarize_by_speaker"] and meeting_info:
            try:
                system_prompt = system_prompt.format(**meeting_info)
            except KeyError as e:
                logger.error("Missing key in meeting_info for prompt formatting: {e}")
                pass

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2
            )

            raw_response_text = response.choices[0].message.content.strip()
            
            cleaned_response = clean_ai_response(raw_response_text)
            
            return cleaned_response

        except OpenAIError as e:
            logger.error(f"OpenAI API error during task '{task}': {e}")
            raise  


ai_service = AIService()

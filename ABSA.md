**research journey #1**

# Model implementation plan
จากการที่ได้คุยกับ อ. เมื่อวันเสาร์ที่ผ่านมา ทำให้ลองสรุป แนวทางการ implement model ออกเป็น 4 แนวทางหลักๆ ดังนี้
* [Option A - base model ใช้ wangchanberta เป็น encoder แล้ว fine tuning ในส่วน ABSA เพิ่มเติม](#Option-A)
* [Option B - base model ใช้ SCB/typhoon หรือ Pathumma เป็น LLM และใช้ความสามารถของ generative ai ในการ วิเคราะห์ ABSA ผ่าน prompt](#Option-B)
* [Option C - base model ใช้ SCB/typhoon หรือ Pathumma เป็น LLM และทำการ fine tuning เพิ่มเติมผ่านทาง LoRA  (Low-Rank Adaptation)](#Option-C)
* [Option D -  แนวทางการใช้ XGBoost เข้ามาวิเคราะห์ แทนที่ตัว Language Model](#Option-D)




## Option A
 base model ใช้ wangchanberta เป็น encoder แล้ว fine tuning ในส่วน ABSA เพิ่มเติม เป็นแนวคิดเริ่มแรก และได้ลองเขียน code ไปบ้างแล้ว  
 
### Pros:
* น่าจะเป็นวิธีที่ตรงกับการทำ ABSA ทั่วไปมากที่สุด

* ขนาด model ค่อนข้างเล็ก เมื่อเทียบกับวิธีอื่น ง่ายต่อการ train / deployment
* ได้เข้าถึงกระบวนการที่เคยเรียนอย่างเต็มที่ ทั้งการ transfer learning, model trainig , model evaluation
* ได้ผลลัพธ์เป็น model สามารถทำ ML Ops ได้อย่างเต็มรูปแบบ
### Cons:
* wangchanberta เป็น model ที่ออกมาสักพักแล้ว ความสามารถในการเรียนรู้ภาษาจะค่่อนข้าง"เก่า" ไม่สามารถเข้าใจ slang / trendy phrase ต่างๆ เมื่อเทียบกับ LLM ใหม่ๆ 
* <span style="background-color: orange; color: black;">เท่าที่ลองหาข้อมูลดู model มีความจำเพาะกับภาษาค่อนข้างสูง เมื่อเจอ review ในภาษาที่ไม่รู้จัก ผลลัพธ์การทำนายแทบไม่ต่างจากการเดาสุ่ม Gemini แนะนำให้ switch ไปใช้ **PhayaThaiBERT** ซึ่งออกมาแก้ปัญหานี้โดยตรง เลยอยากขอคำแนะนำ ว่าควร train เฉพาะภาษาไทย หรือ (อย่างน้อย)ควรรวมภาษาอังกฤษเข้าไปด้วย </span>

 ปัจจุบัน ได้เขียน code สำหรับ train / save & load model / prediction คร่าวๆได้แล้ว แต่ยังพบปัญหาว่า accuracy ยังแย่อยู่ กำลังแก้ปัญหาโดย
 * การใช้ cirriculum learning คือ แทนที่จะ train ด้วย review text จริง ในการเรียนรู้รอบแรกๆ ให้เริ่มจาก review ที่สั้น กระชับ ไม่ซับซ้อน  เมื่อเห็นว่า model เริ่มจับ pattern ได้ จึงเอา review ที่มีความซับซ้อนมา train เพิ่มเติม ทำให้ model ไม่ overfit อยู่ที่ review ที่สั้นกระชับ
 (ตอนนี้เตรียมไว้ aspect ละ 50  5 x 50 x 3 = 750 review)
 * version นี้ได้ balance จำนวน "nm" ออก โดยจะมีจำนวน "nm" ไม่เกิน จำนวน aspect ที่เจอจริง เช่น
 "อาหารอร่อย ถูกกว่าที่คิด" จะ label ว่า TASTE = pos, PRICE = pos และ random aspect ที่เหลือ (SERVICE/ ATMOSPHERE / TRANSPORT) มาไม่เกิน 2 ตัวและให้ label ว่าเป็น nm  ... โดยรวมจะมี label = nm ไม่เกิน 25% ของ dataset 
 * <span style="background-color: orange; color: black;">ตอนนี้เจอปัญหาว่า chatgpt แนะนำว่าถ้าจะเพิ่ม accuracy ให้เอา "nm"  ออก แล้วใส่เป็น head ใหม่แทน (ทั้งๆที่เคยแนะนำว่าให้เอามารวมกันได้)</span>

* <span style="background-color: orange; color: black;">ตอนนี้พยายามทำเรื่อง ให้ colab notebook ไปอ่าน dataset จาก google map shared drive (แต่เดิมใช้ My Drive รู้สึกว่าข้อมูลกระจัดกระจาย) แต่ยังไม่ work ครับ</span>


* <span style="background-color: orange; color: black;">กำลังเตรียม dataset แบบสั้น กระชับในแต่ละ aspect อยู่ แต่ยังไม่แน่ใจว่ากระชับขนาดไหนถึงเรียกว่าดี / จำนวนที่ต้องใช้ (chatgpt ยังยืนยันที่ประมาณ 1k - 10k) </span> 
## Option B
base model ใช้ scb/typhoon หรือ Pathumma เป็น LLM และใช้ความสามารถของ generative ai ในการ วิเคราะห์ ABSA ผ่าน prompt

### Pros:
* ผลลัพธ์น่าจะมี accuracy สูงมาก ตามขนาด parameters ของ LLM ที่นำมาใช้
### Cons:
* ไม่ได้ prediction model ออกมาเป็นผลลัพธ์ วิธีนี้เป็นแค่เพียงการใช้ ความสามารถของ LLM เพื่อการวิเคราะห์ผ่าน prompt เท่านั้น
* เพื่อรักษา client's data privacy ทำให้ไม่สามารถใช้งานผ่าน 3rd party LLM API ได้ ...ทำให้ต้องมีการติดตั้ง local LLM model ซึ่งมีขนาดใหญ่มาก
* ไม่ได้ใช้งาน ML theories / processes ต่างๆ ที่เรียนมา
* prediction time น่าจะนานที่สุด

## Option C

base model ใช้ SCB/typhoon หรือ Pathumma เป็น LLM และทำการ fine tuning เพิ่มเติมผ่านทาง LoRA  (Low-Rank Adaptation)

อันนี้ได้มาจากการ ถาม chatgpt เรื่องการ finetuning LLM ขนาดใหญ่ๆ เท่าที่อ่านๆดูคือการ train บน layer พิเศษที่ขนาดไม่ใหญ่มาก (หลัก XXX mb) เพื่อเสริมความสามารถเฉพาะทางของ LLM

เคยได้ยินชื่อ LoRA มาจาก Image Generator ว่าเป็นการทำให้ model รู้จักองค์ความรู้ใหม่ๆ เช่น ตัวละครเฉพาะ / สถานที่ที่มีลักษณะเฉพาะ โดยไม่ต้องไปยุ่งกับตัว main image generator model และสามารถ"เติม"ได้มากกว่า 1 LoRA  ประมาณว่าเป็นส่วนเสริมที่เติม"องค์ความรู้" ให้ model หลัก โดยตัว LoRA เองมีขนาดเล็กกว่าตัว model มากๆ

### Pros:
* ผลลัพธ์น่าจะมี accuracy สูงมาก ตามขนาด parameters ของ LLM ที่นำมาใช้
* ได้ prediction model ออกมาเป็นผลลัพธ์  LoRA ที่เกิดจากการ train มีขนาดเล็กกว่ามาก 
* ได้ train เอง, เทคโนโลยีค่อนข้างใหม่กว่า traditional finetuning

### Cons:
* ยังติดเรื่อง client's data privacy ทำให้ต้องมีการติดตั้ง local LLM model ซึ่งมีขนาดใหญ่มาก
* prediction time ยังน่าจะนานตามการ encode ของ LLM

โดยส่วนตัว อันนี้มองเป็นทางเลือกไว้ ถ้ากรณี embedding จาก encoder มี accuracy ไม่สูงพอ แต่ยังไม่ได้ศึกษาวิธี code ว่าทำยากหรือเปล่า

## Option D

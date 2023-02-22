# DeepLearning 4th project
## 차량 객체 탐지에 대한 고찰 - YOLOv5와 Detectron2를 위주로
### 상세 내용
- 실제 고속도로 CCTV 영상 캡쳐본을 라벨링, 증강하여 두 모델에 학습시킨 후 정확도가 높은 모델을 선정했다. 모델의 정확도를 높여 차량의 속도나 밀집도를 관측하여 사고 방지에 활용하거나, 모델을 개량하여 사람의 머릿수를 측정하여 밀집도를 인식함으로써 위험상황을 예방하는 모델로 활용 방안을 제시했다.
### 담당 역할
- 주제 선정
- API를 활용해 실제 고속도로 CCTV 데이터 수집 및 roboflow 라벨링
- YOLOv5 모델 학습
- PPT 제작 및 발표

![a (7)](https://user-images.githubusercontent.com/120777172/211731918-adbcd16e-6ddc-4ab3-9e73-3cde60cdcfda.jpg)
![a (9)](https://user-images.githubusercontent.com/120777172/211731944-2c54f55d-b9b4-4885-aa3b-d09d9ec6d34b.jpg)
![a (10)](https://user-images.githubusercontent.com/120777172/211731954-c20de9e7-4b9c-4774-bce7-345066116d79.jpg)
![a (11)](https://user-images.githubusercontent.com/120777172/211731962-a6bb2b7b-5cfc-434d-9dcf-92125da8b581.jpg)
![a (13)](https://user-images.githubusercontent.com/120777172/211731993-f3d311f0-9fdc-4b05-9321-4f85db5c57c9.jpg)
![a (14)](https://user-images.githubusercontent.com/120777172/211732011-6af16b14-6542-412d-99e5-f96171031164.jpg)
![a (16)](https://user-images.githubusercontent.com/120777172/211732018-2b22dd1c-e106-4f42-b625-69f88696a290.jpg)
![a (17)](https://user-images.githubusercontent.com/120777172/211732028-aede40ff-9830-46dc-98e7-18760f64b794.jpg)
![a (18)](https://user-images.githubusercontent.com/120777172/211732035-dd45145d-b148-467e-ae7b-c78596248db9.jpg)
![a (19)](https://user-images.githubusercontent.com/120777172/211732039-eea78870-e5c8-46b3-bfd7-954e76e7bc5c.jpg)
![a (20)](https://user-images.githubusercontent.com/120777172/211732050-6044e54a-dcda-49e3-bb53-220bd8fcc7b4.jpg)
![a (21)](https://user-images.githubusercontent.com/120777172/211732058-78d29b05-1379-4f29-a426-df88831e1acf.jpg)
![a (22)](https://user-images.githubusercontent.com/120777172/211732072-5aac7106-fd2f-45a6-962e-2a14bd2db4df.jpg)
![a (23)](https://user-images.githubusercontent.com/120777172/211732079-139b2f71-88d4-4a1e-8752-0f85a4e3c63b.jpg)
![a (24)](https://user-images.githubusercontent.com/120777172/211732086-d5096e67-de90-4dee-a0a7-373d1554311d.jpg)
![a (25)](https://user-images.githubusercontent.com/120777172/211732100-3b91c15c-5540-41b1-bdbe-2fccbb50c647.jpg)
![a (26)](https://user-images.githubusercontent.com/120777172/211732106-af123598-5d87-4f63-95ea-63803159ba31.jpg)
![a (27)](https://user-images.githubusercontent.com/120777172/211732126-bb3905e0-2448-4496-a087-c2a0718a159e.jpg)
![a (28)](https://user-images.githubusercontent.com/120777172/211732142-69d236a5-f3a7-43f6-a392-9ead92dc2494.jpg)
![a (29)](https://user-images.githubusercontent.com/120777172/211732156-cfe7c128-4f57-4195-9c9f-673079833cdb.jpg)
![a (30)](https://user-images.githubusercontent.com/120777172/211732167-3580646e-9d22-45e5-ac19-6810dfab9578.jpg)
![a (31)](https://user-images.githubusercontent.com/120777172/211732184-ef25bfc8-34b8-4ba1-92df-1ccaadf898a8.jpg)
![a (32)](https://user-images.githubusercontent.com/120777172/211732191-c479463d-2414-458d-9e02-89e75670d593.jpg)
![a (33)](https://user-images.githubusercontent.com/120777172/211732204-5c0429dc-e4e1-44b9-8f60-2a8ef4e0f9fd.jpg)
![a (34)](https://user-images.githubusercontent.com/120777172/211732211-f0203ee4-e06f-4a38-a881-40eb83131a87.jpg)
![a (35)](https://user-images.githubusercontent.com/120777172/211732224-65fe7366-23f5-48cd-bf4f-0bdc07cf6d05.jpg)
![a (36)](https://user-images.githubusercontent.com/120777172/211732232-160c6233-2556-40dd-a6f1-3d57950d6f03.jpg)
![a (37)](https://user-images.githubusercontent.com/120777172/211732241-b4f2c9ae-6f33-49ed-8a27-14cae1ccbd79.jpg)
![a (38)](https://user-images.githubusercontent.com/120777172/211732251-0c9fb184-85e7-4d7f-93bf-c51c18acee11.jpg)
![a (1)](https://user-images.githubusercontent.com/120777172/211732305-1547c741-8cd3-47f2-8bcc-1ced224e6426.jpg)
![a (2)](https://user-images.githubusercontent.com/120777172/211732315-94a04fdc-1f55-4777-ace0-887ca2c72ad5.jpg)
![a (3)](https://user-images.githubusercontent.com/120777172/211732325-0c3438ff-e343-436e-84a2-a6656443c14b.jpg)
![a (4)](https://user-images.githubusercontent.com/120777172/211732336-0b61121f-a666-4555-ad9c-b6dab0f244d0.jpg)
![a (5)](https://user-images.githubusercontent.com/120777172/211732350-3a656de8-d84e-462b-821f-5aaa898ad59f.jpg)
![a (6)](https://user-images.githubusercontent.com/120777172/211732368-a5dcd973-e6ae-44f0-b862-0bf59d1e97f0.jpg)


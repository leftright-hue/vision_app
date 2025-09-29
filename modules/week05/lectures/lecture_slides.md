# 🎯 Week 5: 객체 탐지 이론과 YOLO 실습

## 📌 학습 목표

이번 주차에서는 컴퓨터 비전의 핵심 태스크인 객체 탐지(Object Detection)의 이론과 실제 구현을 학습합니다.

**핵심 학습 내용:**
- 🔍 객체 탐지의 기본 개념과 평가 지표
- 📈 R-CNN 계열의 발전 과정과 Two-stage 방식
- ⚡ YOLO의 One-stage 방식과 실시간 처리
- 🛠️ YOLOv8을 활용한 커스텀 객체 탐지기 구축

---

## 1. 객체 탐지 개요

### 1.1 객체 탐지란?

#### 정의
> **객체 탐지(Object Detection)**: 이미지에서 관심 있는 객체들을 찾아내고, 각 객체의 위치를 바운딩 박스로 표시하며, 객체의 클래스를 분류하는 작업

#### 이미지 분류 vs 객체 탐지
```python
# 이미지 분류 (Image Classification)
input: 이미지
output: 클래스 라벨 (예: "고양이")

# 객체 탐지 (Object Detection)  
input: 이미지
output: [
    {"class": "고양이", "bbox": [x1, y1, x2, y2], "confidence": 0.95},
    {"class": "개", "bbox": [x3, y3, x4, y4], "confidence": 0.87},
    ...
]
```

#### 객체 탐지의 도전과제
1. **다중 객체**: 하나의 이미지에 여러 객체가 존재
2. **다양한 크기**: 같은 클래스라도 크기가 다양함
3. **가려짐(Occlusion)**: 객체들이 서로 겹쳐있음
4. **배경 복잡성**: 복잡한 배경에서 객체 구분
5. **실시간 처리**: 빠른 추론 속도 요구

### 1.2 객체 탐지의 핵심 구성 요소

#### 1. 바운딩 박스 (Bounding Box)
```python
# 바운딩 박스 표현 방식들
bbox_formats = {
    "xyxy": [x_min, y_min, x_max, y_max],           # 좌상단, 우하단 좌표
    "xywh": [x_center, y_center, width, height],    # 중심점과 크기
    "cxcywh": [cx, cy, w, h],                       # 정규화된 중심점과 크기
}

# 바운딩 박스 변환 함수
def xyxy_to_xywh(bbox):
    """xyxy 형식을 xywh 형식으로 변환"""
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return [x_center, y_center, width, height]

def xywh_to_xyxy(bbox):
    """xywh 형식을 xyxy 형식으로 변환"""
    x_center, y_center, width, height = bbox
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]
```

#### 2. 신뢰도 점수 (Confidence Score)
```python
# 신뢰도 점수 계산
confidence = P(object) * IoU(pred_box, true_box)

# P(object): 해당 위치에 객체가 있을 확률
# IoU: 예측 박스와 실제 박스의 겹침 정도
```

#### 3. 클래스 확률 (Class Probability)
```python
# 각 클래스에 대한 확률 분포
class_probs = softmax([logit_cat, logit_dog, logit_car, ...])
# 예: [0.7, 0.2, 0.05, 0.03, 0.02]
```

### 1.3 평가 지표

#### IoU (Intersection over Union)
```python
def calculate_iou(box1, box2):
    """
    두 바운딩 박스의 IoU 계산
    
    Args:
        box1, box2: [x1, y1, x2, y2] 형식의 바운딩 박스
    
    Returns:
        iou: 0과 1 사이의 IoU 값
    """
    # 교집합 영역 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 교집합 면적
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 각 박스의 면적
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 합집합 면적
    union = area1 + area2 - intersection
    
    # IoU 계산
    iou = intersection / union if union > 0 else 0
    return iou

# IoU 해석
# IoU > 0.5: 일반적으로 "좋은" 탐지로 간주
# IoU > 0.7: "매우 좋은" 탐지
# IoU > 0.9: "거의 완벽한" 탐지
```

#### mAP (mean Average Precision)
```python
def calculate_ap(precisions, recalls):
    """
    Average Precision 계산 (11-point interpolation)
    """
    ap = 0
    for t in np.arange(0, 1.1, 0.1):  # 0.0, 0.1, 0.2, ..., 1.0
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    return ap

def calculate_map(all_aps):
    """
    mean Average Precision 계산
    """
    return np.mean(all_aps)

# mAP 변형들
# mAP@0.5: IoU 임계값 0.5에서의 mAP
# mAP@0.5:0.95: IoU 0.5부터 0.95까지 0.05 간격으로 평균한 mAP
# mAP@small/medium/large: 객체 크기별 mAP
```

---

## 2. R-CNN 계열의 발전사

### 2.1 R-CNN (2014): 객체 탐지의 시작

#### 핵심 아이디어
1. **Region Proposal**: Selective Search로 객체가 있을 만한 영역 제안
2. **CNN Feature Extraction**: 각 영역에서 CNN으로 특징 추출
3. **Classification**: SVM으로 객체 분류

#### R-CNN 구조
```python
class RCNN:
    def __init__(self):
        self.region_proposal = SelectiveSearch()
        self.cnn = AlexNet(pretrained=True)
        self.svm_classifiers = {}  # 클래스별 SVM
        self.bbox_regressors = {}  # 클래스별 바운딩 박스 회귀
    
    def forward(self, image):
        # 1. Region Proposal (약 2000개 영역)
        regions = self.region_proposal(image)
        
        # 2. 각 영역을 227x227로 리사이즈
        resized_regions = [resize(region, (227, 227)) for region in regions]
        
        # 3. CNN으로 특징 추출
        features = []
        for region in resized_regions:
            feature = self.cnn.extract_features(region)  # 4096-dim
            features.append(feature)
        
        # 4. SVM으로 분류
        predictions = []
        for feature in features:
            class_scores = {}
            for class_name, svm in self.svm_classifiers.items():
                score = svm.predict(feature)
                class_scores[class_name] = score
            predictions.append(class_scores)
        
        # 5. 바운딩 박스 회귀
        refined_boxes = []
        for i, (feature, region) in enumerate(zip(features, regions)):
            predicted_class = max(predictions[i], key=predictions[i].get)
            regressor = self.bbox_regressors[predicted_class]
            refined_box = regressor.predict(feature, region)
            refined_boxes.append(refined_box)
        
        return predictions, refined_boxes
```

#### R-CNN의 한계
- **속도**: 이미지당 47초 (GPU 기준)
- **메모리**: 각 영역마다 CNN 연산 필요
- **복잡성**: 3단계 파이프라인 (Region Proposal → CNN → SVM)

### 2.2 Fast R-CNN (2015): 속도 개선

#### 주요 개선사항
1. **전체 이미지 CNN**: 이미지 전체에 한 번만 CNN 적용
2. **RoI Pooling**: 다양한 크기의 영역을 고정 크기로 변환
3. **Multi-task Loss**: 분류와 바운딩 박스 회귀를 동시에 학습

#### Fast R-CNN 구조
```python
class FastRCNN:
    def __init__(self, num_classes):
        self.backbone = VGG16(pretrained=True)
        self.roi_pool = RoIPooling(output_size=(7, 7))
        
        # 분류와 회귀를 위한 헤드
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes + 1)  # +1 for background
        )
        
        self.bbox_regressor = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4 * num_classes)  # 4 coordinates per class
        )
    
    def forward(self, image, rois):
        # 1. 전체 이미지에 CNN 적용
        feature_map = self.backbone(image)  # [1, 512, H/16, W/16]
        
        # 2. RoI Pooling
        roi_features = []
        for roi in rois:
            pooled = self.roi_pool(feature_map, roi)  # [512, 7, 7]
            roi_features.append(pooled.flatten())
        
        roi_features = torch.stack(roi_features)  # [N, 512*7*7]
        
        # 3. 분류 및 바운딩 박스 회귀
        class_scores = self.classifier(roi_features)  # [N, num_classes+1]
        bbox_deltas = self.bbox_regressor(roi_features)  # [N, 4*num_classes]
        
        return class_scores, bbox_deltas

# RoI Pooling 구현
class RoIPooling(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.adaptive_pool = nn.AdaptiveMaxPool2d(output_size)
    
    def forward(self, feature_map, roi):
        # roi: [x1, y1, x2, y2] (feature map 좌표계)
        x1, y1, x2, y2 = roi
        
        # 관심 영역 추출
        roi_feature = feature_map[:, :, y1:y2, x1:x2]
        
        # 고정 크기로 풀링
        pooled = self.adaptive_pool(roi_feature)
        
        return pooled
```

#### 성능 개선
- **속도**: 이미지당 2.3초 (9배 빠름)
- **정확도**: mAP 66% (R-CNN 대비 향상)

### 2.3 Faster R-CNN (2015): End-to-End 학습

#### 혁신적 아이디어: RPN (Region Proposal Network)
```python
class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, num_anchors=9):
        super().__init__()
        
        # 3x3 컨볼루션
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        
        # 분류: 객체/배경
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, 1)
        
        # 회귀: 바운딩 박스 조정
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, 1)
        
        # 앵커 생성기
        self.anchor_generator = AnchorGenerator(
            scales=[8, 16, 32],  # 3개 스케일
            ratios=[0.5, 1.0, 2.0]  # 3개 비율
        )  # 총 9개 앵커 per position
    
    def forward(self, feature_map):
        batch_size, _, H, W = feature_map.shape
        
        # 특징 추출
        x = F.relu(self.conv(feature_map))
        
        # 분류 점수
        cls_logits = self.cls_logits(x)  # [B, 18, H, W]
        cls_logits = cls_logits.view(batch_size, 2, -1)  # [B, 2, H*W*9]
        
        # 바운딩 박스 회귀
        bbox_pred = self.bbox_pred(x)  # [B, 36, H, W]
        bbox_pred = bbox_pred.view(batch_size, 4, -1)  # [B, 4, H*W*9]
        
        # 앵커 생성
        anchors = self.anchor_generator(feature_map.shape[-2:])
        
        return cls_logits, bbox_pred, anchors

class AnchorGenerator:
    def __init__(self, scales, ratios, stride=16):
        self.scales = scales
        self.ratios = ratios
        self.stride = stride
    
    def __call__(self, feature_size):
        H, W = feature_size
        anchors = []
        
        for y in range(H):
            for x in range(W):
                # 특징 맵 좌표를 원본 이미지 좌표로 변환
                center_x = x * self.stride
                center_y = y * self.stride
                
                for scale in self.scales:
                    for ratio in self.ratios:
                        # 앵커 크기 계산
                        w = scale * np.sqrt(ratio)
                        h = scale / np.sqrt(ratio)
                        
                        # 앵커 박스 좌표
                        x1 = center_x - w / 2
                        y1 = center_y - h / 2
                        x2 = center_x + w / 2
                        y2 = center_y + h / 2
                        
                        anchors.append([x1, y1, x2, y2])
        
        return torch.tensor(anchors)
```

#### Faster R-CNN 전체 구조
```python
class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 백본 네트워크
        self.backbone = ResNet50(pretrained=True)
        
        # RPN
        self.rpn = RegionProposalNetwork()
        
        # Fast R-CNN 헤드
        self.roi_head = FastRCNNHead(num_classes)
    
    def forward(self, images, targets=None):
        # 1. 특징 추출
        features = self.backbone(images)
        
        # 2. RPN으로 객체 제안
        rpn_cls, rpn_bbox, anchors = self.rpn(features)
        
        # 3. 제안된 영역 선별 (NMS 적용)
        proposals = self.generate_proposals(rpn_cls, rpn_bbox, anchors)
        
        # 4. Fast R-CNN으로 최종 분류 및 회귀
        if self.training:
            # 학습 시: Ground Truth와 매칭
            proposals = self.assign_targets(proposals, targets)
        
        cls_scores, bbox_preds = self.roi_head(features, proposals)
        
        return cls_scores, bbox_preds, proposals
```

#### 성능 개선
- **속도**: 이미지당 0.2초 (실시간 처리 가능)
- **정확도**: mAP 73.2%
- **End-to-End**: 전체 네트워크를 한 번에 학습

---

## 3. One-stage vs Two-stage Detectors

### 3.1 Two-stage Detectors의 특징

#### 장점
- **높은 정확도**: 두 단계로 나누어 정밀한 탐지
- **안정적 성능**: 다양한 데이터셋에서 일관된 성능
- **작은 객체 탐지**: 작은 객체도 잘 탐지

#### 단점
- **느린 속도**: 두 번의 네트워크 통과 필요
- **복잡한 구조**: RPN + Detection Head
- **메모리 사용량**: 많은 중간 결과 저장

### 3.2 One-stage Detectors의 등장

#### 핵심 아이디어
> "Region Proposal 단계를 없애고 한 번에 객체를 탐지하자!"

#### 대표적인 One-stage Detectors
1. **YOLO (You Only Look Once)**
2. **SSD (Single Shot MultiBox Detector)**
3. **RetinaNet**

### 3.3 YOLO의 혁신

#### YOLO v1 (2016)의 핵심 개념
```python
class YOLOv1(nn.Module):
    def __init__(self, num_classes=20, grid_size=7):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = 2  # 그리드 셀당 예측할 박스 수
        
        # 백본 네트워크 (Darknet-19 기반)
        self.backbone = self.build_backbone()
        
        # 최종 출력 레이어
        # 출력 크기: S×S×(B×5 + C)
        # S=7, B=2, C=20 → 7×7×30
        output_size = grid_size * grid_size * (self.num_boxes * 5 + num_classes)
        self.fc = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, output_size)
        )
    
    def forward(self, x):
        # 백본을 통한 특징 추출
        features = self.backbone(x)  # [B, 1024, 7, 7]
        
        # 평탄화 및 완전연결층
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        
        # 출력 재구성: [B, S, S, (B×5 + C)]
        batch_size = x.size(0)
        output = output.view(batch_size, self.grid_size, self.grid_size, -1)
        
        return output
    
    def decode_predictions(self, predictions):
        """
        YOLO 출력을 바운딩 박스와 클래스 확률로 변환
        """
        batch_size, S, S, _ = predictions.shape
        
        # 출력 분해
        # Box 1: [x, y, w, h, confidence]
        # Box 2: [x, y, w, h, confidence]  
        # Class probabilities: [C1, C2, ..., C20]
        
        boxes = predictions[:, :, :, :self.num_boxes * 5]  # [B, S, S, 10]
        class_probs = predictions[:, :, :, self.num_boxes * 5:]  # [B, S, S, 20]
        
        # 바운딩 박스 좌표 변환 (그리드 상대 좌표 → 절대 좌표)
        decoded_boxes = []
        
        for i in range(S):
            for j in range(S):
                for b in range(self.num_boxes):
                    # 박스 정보 추출
                    box_idx = b * 5
                    x = boxes[:, i, j, box_idx]      # 그리드 셀 내 상대 x
                    y = boxes[:, i, j, box_idx + 1]  # 그리드 셀 내 상대 y
                    w = boxes[:, i, j, box_idx + 2]  # 전체 이미지 대비 너비
                    h = boxes[:, i, j, box_idx + 3]  # 전체 이미지 대비 높이
                    conf = boxes[:, i, j, box_idx + 4]  # 신뢰도
                    
                    # 절대 좌표로 변환
                    center_x = (j + x) / S  # 0~1 범위
                    center_y = (i + y) / S  # 0~1 범위
                    
                    decoded_boxes.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': w,
                        'height': h,
                        'confidence': conf,
                        'grid_i': i,
                        'grid_j': j,
                        'box_id': b
                    })
        
        return decoded_boxes, class_probs
```

#### YOLO Loss Function
```python
def yolo_loss(predictions, targets, lambda_coord=5, lambda_noobj=0.5):
    """
    YOLO v1 손실 함수
    
    Args:
        predictions: [B, S, S, (B×5 + C)] 모델 출력
        targets: Ground truth 정보
        lambda_coord: 좌표 손실 가중치
        lambda_noobj: 객체 없는 셀의 신뢰도 손실 가중치
    """
    
    # 1. 좌표 손실 (Coordinate Loss)
    coord_loss = 0
    for target in targets:
        if target['has_object']:
            # 중심점 손실
            coord_loss += (pred_x - target_x)**2 + (pred_y - target_y)**2
            
            # 크기 손실 (제곱근 사용으로 큰 박스와 작은 박스 균형)
            coord_loss += (sqrt(pred_w) - sqrt(target_w))**2 + (sqrt(pred_h) - sqrt(target_h))**2
    
    # 2. 신뢰도 손실 (Confidence Loss)
    conf_loss_obj = 0    # 객체가 있는 경우
    conf_loss_noobj = 0  # 객체가 없는 경우
    
    for i in range(S):
        for j in range(S):
            for b in range(B):
                if grid_has_object[i][j]:
                    # 객체가 있는 그리드: 신뢰도를 IoU에 맞추기
                    target_conf = calculate_iou(pred_box, true_box)
                    conf_loss_obj += (pred_conf - target_conf)**2
                else:
                    # 객체가 없는 그리드: 신뢰도를 0에 맞추기
                    conf_loss_noobj += pred_conf**2
    
    # 3. 분류 손실 (Classification Loss)
    class_loss = 0
    for target in targets:
        if target['has_object']:
            class_loss += sum((pred_class_prob - target_class)**2)
    
    # 총 손실
    total_loss = (lambda_coord * coord_loss + 
                  conf_loss_obj + 
                  lambda_noobj * conf_loss_noobj + 
                  class_loss)
    
    return total_loss
```

---

## 4. YOLO 아키텍처의 발전

### 4.1 YOLOv2/YOLO9000 (2017)

#### 주요 개선사항
1. **Batch Normalization**: 모든 컨볼루션 레이어에 추가
2. **High Resolution Classifier**: 448×448 해상도로 사전 훈련
3. **Anchor Boxes**: Faster R-CNN의 앵커 개념 도입
4. **Dimension Clusters**: K-means로 최적 앵커 크기 결정

```python
class YOLOv2(nn.Module):
    def __init__(self, num_classes=80, num_anchors=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Darknet-19 백본
        self.backbone = Darknet19()
        
        # 검출 헤드
        self.detection_head = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, num_anchors * (5 + num_classes), 1)
        )
        
        # 앵커 박스 (K-means로 결정된 크기)
        self.anchors = [
            (0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
            (7.88282, 3.52778), (9.77052, 9.16828)
        ]
    
    def forward(self, x):
        # 특징 추출
        features = self.backbone(x)  # [B, 1024, 13, 13]
        
        # 검출 헤드 적용
        output = self.detection_head(features)  # [B, 425, 13, 13]
        
        # 출력 재구성: [B, 13, 13, 5, 85]
        batch_size = x.size(0)
        grid_size = output.size(-1)
        output = output.view(batch_size, self.num_anchors, 
                           5 + self.num_classes, grid_size, grid_size)
        output = output.permute(0, 3, 4, 1, 2)  # [B, 13, 13, 5, 85]
        
        return output

# K-means 클러스터링으로 앵커 크기 결정
def generate_anchors(annotations, num_anchors=5):
    """
    K-means를 사용하여 최적의 앵커 박스 크기 결정
    """
    # 모든 바운딩 박스의 너비, 높이 수집
    boxes = []
    for ann in annotations:
        for bbox in ann['bboxes']:
            w, h = bbox[2], bbox[3]  # 정규화된 너비, 높이
            boxes.append([w, h])
    
    boxes = np.array(boxes)
    
    # K-means 클러스터링
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_anchors, random_state=42)
    kmeans.fit(boxes)
    
    # 클러스터 중심점이 앵커 크기
    anchors = kmeans.cluster_centers_
    
    return anchors.tolist()
```

### 4.2 YOLOv3 (2018): 다중 스케일 검출

#### 핵심 혁신: Feature Pyramid Network (FPN)
```python
class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
        # Darknet-53 백본
        self.backbone = Darknet53()
        
        # 3개 스케일에서 검출
        self.detection_layers = nn.ModuleList([
            self.make_detection_layer(1024, 512),  # 13×13
            self.make_detection_layer(768, 256),   # 26×26  
            self.make_detection_layer(384, 128),   # 52×52
        ])
        
        # 각 스케일별 앵커 (총 9개)
        self.anchors = [
            [(116, 90), (156, 198), (373, 326)],    # 큰 객체용
            [(30, 61), (62, 45), (59, 119)],        # 중간 객체용
            [(10, 13), (16, 30), (33, 23)]          # 작은 객체용
        ]
    
    def make_detection_layer(self, in_channels, mid_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_channels, mid_channels * 2, 3, padding=1),
            nn.BatchNorm2d(mid_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_channels * 2, 3 * (5 + self.num_classes), 1)
        )
    
    def forward(self, x):
        # 백본을 통한 다중 스케일 특징 추출
        features = self.backbone(x)
        
        # features는 3개 스케일의 특징 맵 리스트
        # features[0]: [B, 1024, 13, 13] - 큰 객체용
        # features[1]: [B, 512, 26, 26]  - 중간 객체용  
        # features[2]: [B, 256, 52, 52]  - 작은 객체용
        
        detections = []
        
        for i, (feature, detection_layer) in enumerate(zip(features, self.detection_layers)):
            # 각 스케일에서 검출 수행
            detection = detection_layer(feature)
            
            # 출력 재구성
            batch_size, _, grid_h, grid_w = detection.shape
            detection = detection.view(batch_size, 3, 5 + self.num_classes, grid_h, grid_w)
            detection = detection.permute(0, 3, 4, 1, 2)  # [B, H, W, 3, 85]
            
            detections.append(detection)
        
        return detections

class Darknet53(nn.Module):
    """
    YOLOv3의 백본 네트워크
    ResNet의 잔차 연결을 도입한 Darknet
    """
    def __init__(self):
        super().__init__()
        
        # 초기 레이어들
        self.conv1 = self.conv_bn_leaky(3, 32, 3)
        self.conv2 = self.conv_bn_leaky(32, 64, 3, stride=2)
        
        # 잔차 블록들
        self.res_block1 = self.make_layer(64, 32, 1)
        self.conv3 = self.conv_bn_leaky(64, 128, 3, stride=2)
        self.res_block2 = self.make_layer(128, 64, 2)
        self.conv4 = self.conv_bn_leaky(128, 256, 3, stride=2)
        self.res_block3 = self.make_layer(256, 128, 8)
        self.conv5 = self.conv_bn_leaky(256, 512, 3, stride=2)
        self.res_block4 = self.make_layer(512, 256, 8)
        self.conv6 = self.conv_bn_leaky(512, 1024, 3, stride=2)
        self.res_block5 = self.make_layer(1024, 512, 4)
    
    def conv_bn_leaky(self, in_channels, out_channels, kernel_size, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                     stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def make_layer(self, in_channels, mid_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, mid_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res_block1(x)
        x = self.conv3(x)
        x = self.res_block2(x)
        x = self.conv4(x)
        route1 = self.res_block3(x)  # 52×52 특징 (작은 객체용)
        x = self.conv5(route1)
        route2 = self.res_block4(x)  # 26×26 특징 (중간 객체용)
        x = self.conv6(route2)
        route3 = self.res_block5(x)  # 13×13 특징 (큰 객체용)
        
        return [route3, route2, route1]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.leaky_relu(out)
        
        return out
```

### 4.3 YOLOv4 (2020): 최적화의 집대성

#### 주요 개선사항
1. **CSPDarknet53**: Cross Stage Partial 연결
2. **PANet**: Path Aggregation Network
3. **Mosaic Data Augmentation**: 4개 이미지 조합
4. **CIoU Loss**: Complete IoU 손실 함수

```python
# Mosaic Data Augmentation
def mosaic_augmentation(images, labels, input_size=416):
    """
    4개 이미지를 조합하여 하나의 모자이크 이미지 생성
    """
    # 4개 이미지 선택
    indices = np.random.choice(len(images), 4, replace=False)
    
    # 모자이크 이미지 초기화
    mosaic_img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    mosaic_labels = []
    
    # 중심점 랜덤 선택 (이미지 크기의 0.5~1.5 범위)
    center_x = int(np.random.uniform(0.5, 1.5) * input_size // 2)
    center_y = int(np.random.uniform(0.5, 1.5) * input_size // 2)
    
    for i, idx in enumerate(indices):
        img = images[idx]
        label = labels[idx]
        
        # 각 이미지를 적절한 크기로 리사이즈
        h, w = img.shape[:2]
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 4개 영역 중 하나에 배치
        if i == 0:  # 좌상단
            x1, y1 = max(center_x - new_w, 0), max(center_y - new_h, 0)
            x2, y2 = center_x, center_y
        elif i == 1:  # 우상단
            x1, y1 = center_x, max(center_y - new_h, 0)
            x2, y2 = min(center_x + new_w, input_size), center_y
        elif i == 2:  # 좌하단
            x1, y1 = max(center_x - new_w, 0), center_y
            x2, y2 = center_x, min(center_y + new_h, input_size)
        else:  # 우하단
            x1, y1 = center_x, center_y
            x2, y2 = min(center_x + new_w, input_size), min(center_y + new_h, input_size)
        
        # 이미지 배치
        mosaic_img[y1:y2, x1:x2] = img_resized[:y2-y1, :x2-x1]
        
        # 라벨 좌표 조정
        for bbox in label:
            # 원본 좌표를 모자이크 좌표로 변환
            bbox_x1 = bbox[0] * new_w + x1
            bbox_y1 = bbox[1] * new_h + y1
            bbox_x2 = bbox[2] * new_w + x1
            bbox_y2 = bbox[3] * new_h + y1
            
            # 클리핑
            bbox_x1 = max(0, min(bbox_x1, input_size))
            bbox_y1 = max(0, min(bbox_y1, input_size))
            bbox_x2 = max(0, min(bbox_x2, input_size))
            bbox_y2 = max(0, min(bbox_y2, input_size))
            
            # 유효한 박스만 추가
            if bbox_x2 > bbox_x1 and bbox_y2 > bbox_y1:
                mosaic_labels.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox[4]])
    
    return mosaic_img, mosaic_labels

# CIoU Loss
def ciou_loss(pred_boxes, target_boxes):
    """
    Complete IoU Loss
    IoU + 중심점 거리 + 종횡비 일관성을 모두 고려
    """
    # 기본 IoU 계산
    iou = calculate_iou(pred_boxes, target_boxes)
    
    # 중심점 거리
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    
    center_distance = ((pred_center_x - target_center_x) ** 2 + 
                      (pred_center_y - target_center_y) ** 2)
    
    # 대각선 거리
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]
    
    diagonal_distance = ((torch.max(pred_boxes[:, 2], target_boxes[:, 2]) - 
                         torch.min(pred_boxes[:, 0], target_boxes[:, 0])) ** 2 +
                        (torch.max(pred_boxes[:, 3], target_boxes[:, 3]) - 
                         torch.min(pred_boxes[:, 1], target_boxes[:, 1])) ** 2)
    
    # 종횡비 일관성
    v = (4 / (np.pi ** 2)) * torch.pow(
        torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2)
    
    alpha = v / (1 - iou + v + 1e-8)
    
    # CIoU 계산
    ciou = iou - center_distance / diagonal_distance - alpha * v
    
    # Loss (1 - CIoU)
    loss = 1 - ciou
    
    return loss.mean()
```

### 4.4 YOLOv5 (2020): 실용성 강화

#### 주요 특징
1. **PyTorch 구현**: 사용하기 쉬운 PyTorch 기반
2. **AutoAnchor**: 자동 앵커 최적화
3. **Model Scaling**: 다양한 크기의 모델 (n, s, m, l, x)
4. **TTA (Test Time Augmentation)**: 추론 시 증강

```python
# YOLOv5 모델 스케일링
class YOLOv5:
    def __init__(self, model_size='s'):
        self.model_configs = {
            'n': {'depth': 0.33, 'width': 0.25},  # nano
            's': {'depth': 0.33, 'width': 0.50},  # small
            'm': {'depth': 0.67, 'width': 0.75},  # medium
            'l': {'depth': 1.00, 'width': 1.00},  # large
            'x': {'depth': 1.33, 'width': 1.25},  # xlarge
        }
        
        config = self.model_configs[model_size]
        self.depth_multiple = config['depth']
        self.width_multiple = config['width']
    
    def scale_model(self, base_channels, base_depth):
        """모델 크기에 따른 채널 수와 깊이 조정"""
        scaled_channels = int(base_channels * self.width_multiple)
        scaled_depth = max(1, int(base_depth * self.depth_multiple))
        
        return scaled_channels, scaled_depth

# AutoAnchor
class AutoAnchor:
    def __init__(self, dataset, num_anchors=9, img_size=640):
        self.dataset = dataset
        self.num_anchors = num_anchors
        self.img_size = img_size
    
    def generate_anchors(self):
        """데이터셋 분석을 통한 자동 앵커 생성"""
        # 모든 바운딩 박스 수집
        boxes = []
        for data in self.dataset:
            for bbox in data['bboxes']:
                w = bbox[2] * self.img_size  # 절대 크기로 변환
                h = bbox[3] * self.img_size
                boxes.append([w, h])
        
        boxes = np.array(boxes)
        
        # K-means++ 클러스터링
        anchors = self.kmeans_anchors(boxes, self.num_anchors)
        
        # 3개 스케일로 분할
        anchors = anchors[np.argsort(anchors.prod(1))]  # 면적 기준 정렬
        
        return {
            'small': anchors[:3],    # 작은 객체용
            'medium': anchors[3:6],  # 중간 객체용
            'large': anchors[6:9],   # 큰 객체용
        }
    
    def kmeans_anchors(self, boxes, k):
        """K-means를 사용한 앵커 클러스터링"""
        from sklearn.cluster import KMeans
        
        # 정규화 (0-1 범위)
        boxes_norm = boxes / self.img_size
        
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(boxes_norm)
        
        # 원래 크기로 복원
        anchors = kmeans.cluster_centers_ * self.img_size
        
        return anchors
```

### 4.5 YOLOv8 (2023): 최신 기술 집약

#### 혁신적 개선사항
1. **Anchor-Free**: 앵커 박스 없이 직접 예측
2. **Decoupled Head**: 분류와 회귀 헤드 분리
3. **New Backbone**: CSPDarknet → C2f 모듈
4. **Advanced Augmentation**: MixUp, CutMix 등

```python
class YOLOv8Head(nn.Module):
    """
    YOLOv8의 Decoupled Head
    분류와 회귀를 별도 브랜치에서 처리
    """
    def __init__(self, num_classes, in_channels, num_layers=2):
        super().__init__()
        self.num_classes = num_classes
        
        # 분류 브랜치
        cls_layers = []
        for i in range(num_layers):
            cls_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True)
            ])
        cls_layers.append(nn.Conv2d(in_channels, num_classes, 1))
        self.cls_head = nn.Sequential(*cls_layers)
        
        # 회귀 브랜치 (바운딩 박스 + 객체성)
        reg_layers = []
        for i in range(num_layers):
            reg_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True)
            ])
        reg_layers.append(nn.Conv2d(in_channels, 4 + 1, 1))  # 4(bbox) + 1(objectness)
        self.reg_head = nn.Sequential(*reg_layers)
    
    def forward(self, x):
        # 분류 예측
        cls_output = self.cls_head(x)  # [B, num_classes, H, W]
        
        # 회귀 예측
        reg_output = self.reg_head(x)  # [B, 5, H, W]
        
        return cls_output, reg_output

# Anchor-Free 예측 디코딩
def decode_yolov8_predictions(cls_output, reg_output, stride):
    """
    YOLOv8 앵커 프리 예측 디코딩
    """
    batch_size, num_classes, H, W = cls_output.shape
    
    # 그리드 생성
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [H, W, 2]
    
    # 회귀 출력 분해
    bbox_pred = reg_output[:, :4]      # [B, 4, H, W] - 바운딩 박스
    obj_pred = reg_output[:, 4:5]      # [B, 1, H, W] - 객체성
    
    # 바운딩 박스 디코딩 (ltrb → xyxy)
    bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # [B, H, W, 4]
    
    # 거리 기반 예측을 절대 좌표로 변환
    lt = grid.unsqueeze(0) - bbox_pred[..., :2]  # left, top
    rb = grid.unsqueeze(0) + bbox_pred[..., 2:]  # right, bottom
    
    # 최종 바운딩 박스 (픽셀 좌표)
    bbox_final = torch.cat([lt, rb], dim=-1) * stride
    
    # 클래스 확률과 객체성 결합
    cls_prob = torch.sigmoid(cls_output).permute(0, 2, 3, 1)  # [B, H, W, num_classes]
    obj_prob = torch.sigmoid(obj_pred).permute(0, 2, 3, 1)   # [B, H, W, 1]
    
    # 최종 신뢰도 = 클래스 확률 × 객체성
    confidence = cls_prob * obj_prob
    
    return bbox_final, confidence
```

---

## 5. NMS (Non-Maximum Suppression)

### 5.1 NMS의 필요성

#### 문제점: 중복 검출
```python
# 객체 탐지 결과 예시 (중복 검출)
detections = [
    {"bbox": [100, 100, 200, 200], "class": "person", "confidence": 0.9},
    {"bbox": [105, 95, 205, 195], "class": "person", "confidence": 0.85},
    {"bbox": [98, 102, 198, 202], "class": "person", "confidence": 0.8},
    {"bbox": [300, 150, 400, 250], "class": "car", "confidence": 0.95},
]
# → 같은 사람을 3번 검출!
```

### 5.2 NMS 알고리즘

```python
def non_max_suppression(detections, iou_threshold=0.5, conf_threshold=0.5):
    """
    Non-Maximum Suppression 구현
    
    Args:
        detections: 검출 결과 리스트
        iou_threshold: IoU 임계값 (겹침 정도)
        conf_threshold: 신뢰도 임계값
    
    Returns:
        filtered_detections: NMS 적용 후 결과
    """
    
    # 1. 신뢰도 임계값 이하 제거
    detections = [det for det in detections if det['confidence'] >= conf_threshold]
    
    if not detections:
        return []
    
    # 2. 신뢰도 기준 내림차순 정렬
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 3. 클래스별로 NMS 적용
    final_detections = []
    classes = set(det['class'] for det in detections)
    
    for class_name in classes:
        # 해당 클래스의 검출 결과만 추출
        class_detections = [det for det in detections if det['class'] == class_name]
        
        # NMS 적용
        keep = []
        while class_detections:
            # 가장 높은 신뢰도의 검출 결과 선택
            best = class_detections.pop(0)
            keep.append(best)
            
            # 나머지와 IoU 계산하여 겹치는 것들 제거
            remaining = []
            for det in class_detections:
                iou = calculate_iou(best['bbox'], det['bbox'])
                if iou <= iou_threshold:
                    remaining.append(det)
            
            class_detections = remaining
        
        final_detections.extend(keep)
    
    return final_detections

# 사용 예시
filtered_results = non_max_suppression(
    detections, 
    iou_threshold=0.5,
    conf_threshold=0.5
)
print(f"원본: {len(detections)}개 → NMS 후: {len(filtered_results)}개")
```

### 5.3 고급 NMS 기법

#### Soft NMS
```python
def soft_nms(detections, sigma=0.5, iou_threshold=0.3, score_threshold=0.001):
    """
    Soft NMS: 겹치는 박스의 점수를 0으로 만들지 않고 감소시킴
    """
    detections = detections.copy()
    
    for i in range(len(detections)):
        if detections[i]['confidence'] < score_threshold:
            continue
            
        for j in range(i + 1, len(detections)):
            if detections[i]['class'] != detections[j]['class']:
                continue
                
            iou = calculate_iou(detections[i]['bbox'], detections[j]['bbox'])
            
            if iou > iou_threshold:
                # 가우시안 가중치 적용
                weight = np.exp(-(iou ** 2) / sigma)
                detections[j]['confidence'] *= weight
    
    # 최종 임계값 이하 제거
    return [det for det in detections if det['confidence'] >= score_threshold]

# DIoU-NMS (Distance-IoU NMS)
def diou_nms(detections, iou_threshold=0.5):
    """
    DIoU를 사용한 NMS (중심점 거리도 고려)
    """
    def calculate_diou(box1, box2):
        # 기본 IoU
        iou = calculate_iou(box1, box2)
        
        # 중심점 거리
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2
        
        center_distance = ((center1_x - center2_x) ** 2 + 
                          (center1_y - center2_y) ** 2)
        
        # 대각선 거리
        diagonal_distance = ((max(box1[2], box2[2]) - min(box1[0], box2[0])) ** 2 +
                           (max(box1[3], box2[3]) - min(box1[1], box2[1])) ** 2)
        
        # DIoU
        diou = iou - center_distance / diagonal_distance
        
        return diou
    
    # DIoU를 사용하여 NMS 적용
    # (구현은 기본 NMS와 유사하지만 IoU 대신 DIoU 사용)
    pass
```

---

## 6. 실습 프로젝트: 교실 물건 탐지기

### 6.1 프로젝트 개요

#### 목표
교실에서 흔히 볼 수 있는 5가지 물건을 탐지하는 커스텀 YOLO 모델 구축

#### 탐지 대상 클래스
1. **책 (Book)**
2. **노트북 (Laptop)**
3. **의자 (Chair)**
4. **칠판 (Whiteboard)**
5. **가방 (Bag)**

### 6.2 데이터셋 준비 전략

#### Roboflow를 활용한 데이터 수집
```python
# 데이터 수집 계획
data_collection_plan = {
    "sources": [
        "직접 촬영 (교실, 도서관, 카페)",
        "오픈 데이터셋 (Open Images, COCO)",
        "웹 크롤링 (저작권 주의)",
        "데이터 증강"
    ],
    
    "target_counts": {
        "book": 500,
        "laptop": 400,
        "chair": 600,
        "whiteboard": 300,
        "bag": 450
    },
    
    "annotation_guidelines": {
        "bbox_quality": "객체 전체를 포함하되 여백 최소화",
        "occlusion": "50% 이상 가려진 객체는 제외",
        "size_limit": "이미지 크기의 1% 이상인 객체만 포함",
        "edge_cases": "부분적으로 잘린 객체도 포함"
    }
}
```

#### 데이터 증강 전략
```python
import albumentations as A

def create_augmentation_pipeline():
    """
    교실 물건 탐지를 위한 데이터 증강 파이프라인
    """
    return A.Compose([
        # 기하학적 변환
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.5
        ),
        
        # 색상 변환
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        
        # 노이즈 및 블러
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
        # 조명 변화
        A.RandomShadow(p=0.3),
        A.RandomSunFlare(p=0.2),
        
        # 컷아웃
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            p=0.3
        ),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))

# 사용 예시
augmentation = create_augmentation_pipeline()

def augment_image(image, bboxes, class_labels):
    """이미지와 바운딩 박스에 증강 적용"""
    augmented = augmentation(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels
    )
    
    return augmented['image'], augmented['bboxes'], augmented['class_labels']
```

### 6.3 YOLOv8 커스텀 학습

#### 데이터셋 구성
```yaml
# dataset.yaml
path: ./classroom_objects  # 데이터셋 루트 경로
train: images/train
val: images/val
test: images/test

# 클래스 정의
nc: 5  # 클래스 수
names: ['book', 'laptop', 'chair', 'whiteboard', 'bag']
```

#### 학습 스크립트
```python
from ultralytics import YOLO
import torch

def train_classroom_detector():
    """
    교실 물건 탐지기 학습
    """
    
    # 1. 사전훈련된 YOLOv8 모델 로드
    model = YOLO('yolov8n.pt')  # nano 버전 (빠른 학습)
    
    # 2. 학습 설정
    training_config = {
        'data': 'dataset.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'lr0': 0.01,
        'weight_decay': 0.0005,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'project': 'classroom_detector',
        'name': 'yolov8n_classroom',
        'save_period': 10,
        'patience': 20,
        'save': True,
        'plots': True,
        'val': True,
    }
    
    # 3. 학습 실행
    results = model.train(**training_config)
    
    # 4. 최고 모델 저장
    best_model = YOLO('runs/detect/yolov8n_classroom/weights/best.pt')
    
    return best_model, results

# 학습 실행
if __name__ == "__main__":
    model, results = train_classroom_detector()
    print("학습 완료!")
    print(f"최고 mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
```

#### 하이퍼파라미터 튜닝
```python
def hyperparameter_tuning():
    """
    Ray Tune을 사용한 하이퍼파라미터 최적화
    """
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    
    def objective(config):
        model = YOLO('yolov8n.pt')
        
        results = model.train(
            data='dataset.yaml',
            epochs=50,
            imgsz=640,
            batch=config['batch'],
            lr0=config['lr0'],
            weight_decay=config['weight_decay'],
            mosaic=config['mosaic'],
            verbose=False
        )
        
        # mAP50을 최대화
        return {"mAP50": results.results_dict['metrics/mAP50(B)']}
    
    # 탐색 공간 정의
    search_space = {
        'batch': tune.choice([8, 16, 32]),
        'lr0': tune.loguniform(1e-4, 1e-1),
        'weight_decay': tune.loguniform(1e-5, 1e-2),
        'mosaic': tune.uniform(0.5, 1.0),
    }
    
    # 스케줄러 설정
    scheduler = ASHAScheduler(
        metric="mAP50",
        mode="max",
        max_t=50,
        grace_period=10,
        reduction_factor=2
    )
    
    # 튜닝 실행
    analysis = tune.run(
        objective,
        config=search_space,
        num_samples=20,
        scheduler=scheduler,
        resources_per_trial={"cpu": 2, "gpu": 0.5}
    )
    
    # 최적 하이퍼파라미터
    best_config = analysis.best_config
    print("최적 하이퍼파라미터:", best_config)
    
    return best_config
```

### 6.4 모델 평가 및 분석

#### 상세 평가 메트릭
```python
def evaluate_model(model, test_dataset):
    """
    모델 성능 상세 평가
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    
    # 1. 기본 메트릭 계산
    results = model.val(data='dataset.yaml', split='test')
    
    # 2. 클래스별 성능 분석
    class_names = ['book', 'laptop', 'chair', 'whiteboard', 'bag']
    
    metrics_summary = {
        'overall': {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
        },
        'per_class': {}
    }
    
    for i, class_name in enumerate(class_names):
        metrics_summary['per_class'][class_name] = {
            'AP50': results.box.ap50[i],
            'AP50-95': results.box.ap[i],
            'precision': results.box.p[i],
            'recall': results.box.r[i],
        }
    
    # 3. 혼동 행렬 생성
    predictions = []
    ground_truths = []
    
    for image_path in test_dataset:
        pred = model.predict(image_path, verbose=False)[0]
        # ... 예측 결과와 실제 라벨 수집
    
    # 4. 시각화
    plot_evaluation_results(metrics_summary, predictions, ground_truths)
    
    return metrics_summary

def plot_evaluation_results(metrics, predictions, ground_truths):
    """평가 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 클래스별 AP50 비교
    classes = list(metrics['per_class'].keys())
    ap50_values = [metrics['per_class'][cls]['AP50'] for cls in classes]
    
    axes[0, 0].bar(classes, ap50_values, alpha=0.7)
    axes[0, 0].set_title('클래스별 AP50')
    axes[0, 0].set_ylabel('AP50')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Precision-Recall 곡선
    # ... PR 곡선 그리기
    
    # 혼동 행렬
    cm = confusion_matrix(ground_truths, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[1, 0])
    axes[1, 0].set_title('혼동 행렬')
    
    # 검출 예시 이미지
    # ... 검출 결과 예시 표시
    
    plt.tight_layout()
    plt.show()
```

---

## 7. 실시간 추론 및 최적화

### 7.1 모델 최적화 기법

#### TensorRT 최적화
```python
def optimize_with_tensorrt(model_path, input_shape=(1, 3, 640, 640)):
    """
    TensorRT를 사용한 모델 최적화
    """
    import tensorrt as trt
    import pycuda.driver as cuda
    
    # ONNX로 변환
    model = YOLO(model_path)
    model.export(format='onnx', imgsz=640)
    
    # TensorRT 엔진 빌드
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # ONNX 파일 파싱
    with open(model_path.replace('.pt', '.onnx'), 'rb') as model_file:
        parser.parse(model_file.read())
    
    # 빌더 설정
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)  # FP16 정밀도
    
    # 엔진 빌드
    engine = builder.build_engine(network, config)
    
    # 엔진 저장
    with open(model_path.replace('.pt', '.trt'), 'wb') as f:
        f.write(engine.serialize())
    
    print("TensorRT 최적화 완료!")
    return model_path.replace('.pt', '.trt')

# 양자화 (Quantization)
def quantize_model(model_path):
    """
    모델 양자화 (INT8)
    """
    model = YOLO(model_path)
    
    # INT8 양자화 설정
    model.export(
        format='onnx',
        int8=True,
        data='dataset.yaml'  # 캘리브레이션 데이터
    )
    
    print("모델 양자화 완료!")
```

#### 모바일 최적화
```python
def optimize_for_mobile(model_path):
    """
    모바일 디바이스용 최적화
    """
    model = YOLO(model_path)
    
    # CoreML 변환 (iOS)
    model.export(format='coreml', nms=True)
    
    # TensorFlow Lite 변환 (Android)
    model.export(format='tflite', int8=True)
    
    # NCNN 변환 (경량화)
    model.export(format='ncnn')
    
    print("모바일 최적화 완료!")
```

### 7.2 실시간 비디오 처리

```python
import cv2
import time
from collections import deque

class RealTimeDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 성능 모니터링
        self.fps_queue = deque(maxlen=30)
        self.detection_history = deque(maxlen=100)
    
    def process_video(self, source=0, output_path=None):
        """
        실시간 비디오 처리
        
        Args:
            source: 비디오 소스 (0=웹캠, 파일경로, RTSP URL)
            output_path: 출력 비디오 저장 경로
        """
        cap = cv2.VideoCapture(source)
        
        # 비디오 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 출력 비디오 설정
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("실시간 탐지 시작... (ESC로 종료)")
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # 객체 탐지
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]
            
            # 결과 시각화
            annotated_frame = self.draw_detections(frame, results)
            
            # FPS 계산
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            self.fps_queue.append(fps)
            avg_fps = sum(self.fps_queue) / len(self.fps_queue)
            
            # 정보 표시
            self.draw_info(annotated_frame, avg_fps, results)
            
            # 화면 출력
            cv2.imshow('교실 물건 탐지기', annotated_frame)
            
            # 비디오 저장
            if output_path:
                out.write(annotated_frame)
            
            # 종료 조건
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
                break
        
        # 정리
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"평균 FPS: {avg_fps:.2f}")
    
    def draw_detections(self, frame, results):
        """검출 결과를 프레임에 그리기"""
        annotated_frame = frame.copy()
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            class_names = ['book', 'laptop', 'chair', 'whiteboard', 'bag']
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                
                # 바운딩 박스 그리기
                color = colors[class_id % len(colors)]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # 라벨 그리기
                label = f"{class_names[class_id]}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def draw_info(self, frame, fps, results):
        """정보 패널 그리기"""
        # FPS 표시
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 검출 개수 표시
        num_detections = len(results.boxes) if results.boxes is not None else 0
        cv2.putText(frame, f"Objects: {num_detections}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 모델 정보
        cv2.putText(frame, "Classroom Object Detector", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# 사용 예시
if __name__ == "__main__":
    detector = RealTimeDetector('best.pt')
    
    # 웹캠으로 실시간 탐지
    detector.process_video(source=0)
    
    # 비디오 파일 처리
    # detector.process_video(source='input_video.mp4', output_path='output_video.mp4')
```

---

## 📚 참고 자료 및 추가 학습

### 논문 및 문서
- **R-CNN**: "Rich feature hierarchies for accurate object detection" (Girshick et al., 2014)
- **Fast R-CNN**: "Fast R-CNN" (Girshick, 2015)
- **Faster R-CNN**: "Faster R-CNN: Towards Real-Time Object Detection" (Ren et al., 2015)
- **YOLO v1**: "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al., 2016)
- **YOLOv8**: "YOLOv8: A New State-of-the-Art for Object Detection" (Ultralytics, 2023)

### 실습 도구 및 플랫폼
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/) - 데이터셋 관리 및 증강
- [Google Colab](https://colab.research.google.com/) - 무료 GPU 학습 환경
- [Weights & Biases](https://wandb.ai/) - 실험 추적 및 시각화

### 데이터셋
- [COCO Dataset](https://cocodataset.org/) - 대규모 객체 탐지 데이터셋
- [Open Images](https://storage.googleapis.com/openimages/web/index.html) - 구글의 오픈 이미지 데이터셋
- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) - 전통적인 객체 탐지 벤치마크

---

## 🎯 이번 주차 핵심 정리

### 학습 완료 체크리스트

✅ **객체 탐지 기초 개념**
- 바운딩 박스, 신뢰도, 클래스 확률
- IoU, mAP 등 평가 지표
- NMS 알고리즘 이해

✅ **R-CNN 계열 발전사**
- R-CNN → Fast R-CNN → Faster R-CNN
- Two-stage 방식의 장단점
- RPN과 앵커 박스 개념

✅ **YOLO 아키텍처**
- One-stage 방식의 혁신
- YOLOv1부터 YOLOv8까지의 발전
- Anchor-free 방식의 이해

✅ **실전 구현 능력**
- YOLOv8을 활용한 커스텀 학습
- 데이터셋 준비 및 증강
- 실시간 객체 탐지 시스템 구축

**🚀 이제 여러분은 객체 탐지의 이론부터 실제 구현까지 완전히 마스터했습니다!**

다음 주에는 이러한 지식을 바탕으로 객체 탐지를 더욱 심화하고, 웹 서비스로 배포하는 방법을 학습하겠습니다.

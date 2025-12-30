# fall_or_fight

이 폴더는 **YOLO8n-Pose** 모델을 사용하여 **낙상(Fall_Down)**과 **싸움(Fight)**을 식별하는 모델 학습 결과를 포함합니다.

## runs 폴더
**runs** 폴더는 **정상(Normal)**, **낙상(Fall_Down)**, **싸움(Fight)**을 식별하는 **YOLO8n-Pose** 모델의 학습 결과를 포함합니다. 이 폴더 안에는 모델의 **학습 결과**와 **기타 출력 파일들**이 저장됩니다.

## discard 폴더
**discard** 폴더는 **기대값과 다른 학습 결과**가 포함된 모델들을 저장합니다. 이 폴더는 **학습 중간 결과**나 **실험적인 모델들**이 저장되어, 후속 작업에서 참고하거나 다시 평가할 수 있도록 합니다.

---
- **Normal**: 정상적인 상태를 식별하는 모델
- **Fall_Down**: 낙상 상태를 식별하는 모델
- **Fight**: 싸움 상태를 식별하는 모델

```
[ 모델별 폐기 사유]

- cctv1
  - Fight 클래스만 인식
  - 그림자를 사람으로 오인식

- cctv2
  - Normal 클래스만 인식
  - 클래스 분류 편향 발생

- cctv_1_stable
  - 라벨링 오류 확인
  - 데이터 품질 문제로 폐기

- cctv_optimized_s_v2
  - Fight 클래스만 인식
  - 그림자를 사람으로 오인식

- cctv_pose_final
  - Fight 클래스만 인식
  - 클래스 일반화 실패

- cctv_v2_5_shadow_master
  - Fight 클래스만 인식
  - 그림자를 사람으로 강하게 오인식
  - cctv_v2_shadow_fix 개선 버전이나 문제 미해결

- cctv_v2_shadow_fix
  - Fight 클래스만 인식
  - 그림자를 사람으로 오인식
  - cctv_optimized_s_v2 개선 시도 버전
```

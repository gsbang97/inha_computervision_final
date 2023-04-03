
import cv2
import numpy as np

# 동영상 파일 열기
cap = cv2.VideoCapture('final.mp4')

# 샷 번호 및 배경 이미지 저장을 위한 변수 초기화
shot_num = 1
bg_img = None

# 이전 프레임에서 사용할 히스토그램 초기화
prev_hist = None

# 히스토그램 차이를 계산하기 위한 상수
alpha = 50000

# 샷 번호 및 이동 방향 저장을 위한 변수 초기화
shot_num = 1

changed = 0
frame_interval = 5 
accumulated_flow = None
frame_count = 0
ret, frame = cap.read()
# background = frame.copy().astype(np.float32)
# background_subtractor = cv2.createBackgroundSubtractorMOG2()
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
bg_alpha = 0.01  # Controls the learning rate for updating the background
prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def get_contours(fg_mask, min_area=500):
    # contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return valid_contours

def draw_total_flow(img, flow, x,y,w,h):
    total_flow_x = np.sum(flow[:, :, 0])
    total_flow_y = np.sum(flow[:, :, 1])

    # h, w = img.shape[:2]
    # center = (w // 2, h // 2)
    center = (x + w//2, y+ h//2)
    new_center = (center[0] + int(total_flow_x)//100, center[1] + int(total_flow_y)//100)

    cv2.arrowedLine(img, center, new_center, (0, 0, 255), 2, tipLength=0.2)

    return img

while cap.isOpened():
    # 동영상에서 프레임 읽어오기
    ret, frame = cap.read()
    


    if ret:
        if changed > 0:
            changed += 1
            cv2.putText(frame, "Shot Changed", text_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if changed == 10:
                changed = 0




        # 현재 프레임에서 히스토그램 계산
        curr_hist = cv2.calcHist([frame], [0], None, [256], [0,256])

        if prev_hist is not None:
            # 히스토그램 차이 계산
            hist_diff = np.abs(curr_hist - prev_hist).sum()
            # 히스토그램 차이가 alpha 이상인 경우, 새로운 샷이 시작된 것으로 판단
            if hist_diff > alpha:
                text_org = (frame.shape[1]-250, 50)
                cv2.putText(frame, "Shot Changed", text_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                changed += 1
                shot_num += 1
                output_image_path = 'background_image'+str(shot_num)+'.png'
                background = background_subtractor.getBackgroundImage()
                cv2.imwrite(output_image_path, background)
                # background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16,
                                                                        #    detectShadows=False)
                background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
                # background = frame.copy().astype(np.float32)
                # prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                # Update the background model
                # fg_mask = background_subtractor.apply(frame)
                # result = draw_contours(frame, fg_mask)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fg_mask = background_subtractor.apply(frame_gray)

                contours = get_contours(fg_mask)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    prev_roi_gray = prev_frame_gray[y:y+h, x:x+w]
                    roi_gray = frame_gray[y:y+h, x:x+w]

                    if prev_roi_gray.shape == roi_gray.shape:
                        flow = cv2.calcOpticalFlowFarneback(prev_roi_gray, roi_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        frame = draw_total_flow(frame, flow,x,y,w,h)
                        # frame[y:y+h, x:x+w] = draw_total_flow(frame[y:y+h, x:x+w], flow)


                
                # cv2.accumulateWeighted(frame, background, bg_alpha)

        # 이전 프레임의 히스토그램 저장
        prev_hist = curr_hist.copy()
        prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1
        # 현재 프레임을 화면에 출력
        # cv2.arrowedLine(frame, (100,100), (200,200), (0, 0, 255), 2, tipLength=0.2)
        cv2.imshow('frame', frame)
        
        # 'q' 키를 누르면 동영상 재생 중지
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# 동영상 파일 닫기
cap.release()
cv2.destroyAllWindows()

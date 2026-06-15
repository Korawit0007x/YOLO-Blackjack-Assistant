from ultralytics import YOLO
import cv2

def blackjack_score(cards):
    total = 0 
    aces = 0
    
    for card in cards:
        rank = card[:-1]
        
        if rank == 'A':
            total += 11
            aces += 1
            
        elif rank in ['J','K','Q']:
            total += 10
        
        else:
            total += int(rank)
            
    while total > 21 and aces >0 :
        total -=10
        aces -=1
        
    return total


model = YOLO("best.pt")

print(model.names)

card_value = {}

detected_cards = set()
 
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    results = model(frame)

    annotated_frame = results[0].plot()
    
    current_frame_cards = set()
    
    total_score = 0
    
    text_color = 0
    
    for box in results[0].boxes:
        cls_num = int(box.cls[0])
        card_name = model.names[cls_num]    
        current_frame_cards.add(card_name)
        
        total_score = blackjack_score(current_frame_cards)
        
        if total_score==21:
            text_color = (0,215,255)
        elif total_score<21:
            text_color = (0,255,0)
        else :
            text_color = (255,0,0)
             
    cv2.putText(
        annotated_frame,
        f'Card : {current_frame_cards}',
        (20,80),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0,255,0),
        2
    )
    cv2.putText(
        annotated_frame,
        f'Total : {total_score}',
        (20,40),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        text_color,
        2
    )
    
    if total_score == 21:
        cv2.putText(
            annotated_frame,
            'BackJack!!',
            (20,120),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0,255,0),
            2
        )
    
    cv2.imshow("YOLO Webcam", annotated_frame)
    
    cv2.imwrite(total_score==21)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print(detected_cards)
        break

cap.release()
cv2.destroyAllWindows()

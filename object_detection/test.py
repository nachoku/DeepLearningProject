from network.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from network.utils.misc import Timer
import cv2
import sys


if len(sys.argv) < 5:
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
image_path = sys.argv[4]

class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
else:
    print("The net type is wrong.")
    sys.exit(1)
net.load(model_path)

if net_type == 'ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
else:
    print("error")

orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)

for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX ,
                1,  # font scale
                (0, 0, 0),
                2)  # line type
path = "output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")

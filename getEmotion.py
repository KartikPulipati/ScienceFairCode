import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

classes = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}
font = cv2.FONT_HERSHEY_SIMPLEX

face_cascade = cv2.CascadeClassifier(
    '/Users/kartik/opt/anaconda3/envs/pytorchenv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
                '/Users/kartik/opt/anaconda3/envs/pytorchenv/share/OpenCV/haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(
                '/Users/kartik/opt/anaconda3/envs/pytorchenv/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')

device = torch.device('cpu')

softmax = torch.nn.Softmax(dim=1)


def img2tensor(x):
    transform = tt.Compose(
                    [tt.ToPILImage(),
                        tt.Grayscale(num_output_channels=1),
                        tt.Resize((48, 48)),
                        tt.ToTensor()])
    return transform(x)


def predict(x):
    out = model(img2tensor(x)[None])
    scaled = softmax(out)
    prob = torch.max(scaled).item()
    label = classes[torch.argmax(scaled).item()]
    return {'label': label, 'probability': prob}

# declaring model
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.input = conv_block(in_channels, 64)

        self.conv1 = conv_block(64, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = conv_block(64, 64, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = conv_block(64, 64, pool=True)
        self.res3 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop3 = nn.Dropout(0.5)

        self.classifier = nn.Sequential(nn.MaxPool2d(6),
                                        nn.Flatten(),
                                        nn.Linear(64, num_classes))

    def forward(self, xb):
        out = self.input(xb)

        out = self.conv1(out)
        out = self.res1(out) + out
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.drop3(out)

        return self.classifier(out)


# loading in model
model = ResNet(1, 7)
model.load_state_dict(torch.load('/Users/kartik/Desktop/ScienceFairCode/stitched_model_state.pth',
                                 map_location=device))

def vConcat_Resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1]
                for img in img_list)

    # resizing images
    im_list_resize = [cv2.resize(img,
                                 (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation=interpolation)
                      for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)
#
frame0 = cv2.VideoCapture(0)
frame1 = cv2.VideoCapture(2)


while 1:
    ret0, img1 = frame0.read()
    ret1, img2 = frame1.read()
    img11 = cv2.resize(img1, (500, 500))
    img22 = cv2.resize(img2, (500, 500))
    ax = []
    ay = []
    ax2 = []
    ay2 = []
    mouths = []

    if frame0:
        gray = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=13)
        for (ex, ey, ew, eh) in eyes:
            img11 = cv2.rectangle(img11, (ex, ey), (ex+ew, ey+eh), (0, 256, 0), 1)
            ax.append(ex)
            ay.append(ey)
            ax2.append(ex+ew)
            ay2.append(ey+eh)
            cv2.imshow("Eye", img11)

    if frame1:
        gray = cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
        mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=100)
        for (mx, my, mw, mh) in mouth:
            img22 = cv2.rectangle(img22, (mx, my), (mx + mw, my + mh), (0, 255, 0), 1)
            mouths.append((mx, my, mx + mw, my + mh))
            cv2.imshow("Mouth", img22)

    if len(mouths) == 1 and len(ay2) == 2:
        eye = img11[min(ay):max(ay2)+10, min(ax)-10:max(ax2)+10]
        mouth = img22[mouths[0][1]-20:mouths[0][3]+20, mouths[0][0]-20:mouths[0][2]+20]
        result = vConcat_Resize([eye, mouth])
        emotion = predict(result)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, result.shape[1]-15)
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 1
        lineType = 2
        cv2.putText(result, emotion['label'],
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        # Display the image
        cv2.imshow("Result", result)
    # else:
    #     img = np.zeros((512, 512, 3), np.uint8)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     bottomLeftCornerOfText = (10, 500)
    #     fontScale = 1
    #     fontColor = (255, 255, 255)
    #     thickness = 1
    #     lineType = 2
    #
    #     cv2.putText(img, 'Reposition Yourself!',
    #                 bottomLeftCornerOfText,
    #                 font,
    #                 fontScale,
    #                 fontColor,
    #                 thickness,
    #                 lineType)
    #
    #     # Display the image
    #     cv2.imshow("Result", img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

frame0.release()
# frame1.release()
cv2.destroyAllWindows()
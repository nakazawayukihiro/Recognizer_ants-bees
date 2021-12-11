# ハチと蟻の画像はPCにダウンロードしました。

# 必要ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# PyTorch関連ライブラリのインポート
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

# warning表示off
import warnings
warnings.simplefilter('ignore')

# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# numpyの表示桁数設定
np.set_printoptions(suppress=True, precision=5)

# デバイスを設定
device = torch.device('cpu')

# イメージとラベル表示(持ってきた)
def show_images_labels(loader, classes, net, device):

    # データローダーから最初の1セットを取得する
    for images, labels in loader:
        break
    # 表示数は50個とバッチサイズのうち小さい方
    n_size = min(len(images), 50)

    if net is not None:
      # デバイスの割り当て
      inputs = images.to(device)
      labels = labels.to(device)

      # 予測計算
      outputs = net(inputs)
      predicted = torch.max(outputs,1)[1]
      #images = images.to('cpu')

    # 最初のn_size個の表示
    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # netがNoneでない場合は、予測結果もタイトルに表示する
        if net is not None:
          predicted_name = classes[predicted[i]]
          # 正解かどうかで色分けをする
          if label_name == predicted_name:
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        # netがNoneの場合は、正解ラベルのみ表示
        else:
          ax.set_title(label_name, fontsize=20)
        # TensorをNumPyに変換
        image_np = images[i].numpy().copy()
        # 軸の順番変更 (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 値の範囲を[-1, 1] -> [0, 1]に戻す
        img = (img + 1)/2
        # 結果表示
        plt.imshow(img)
        ax.set_axis_off()
    plt.show()

# 写真のサイズやテンソル化、正規化
test_transform = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor(),
  transforms.Normalize(0.5, 0.5),
])

# PCにダウンロードした画像群を取り出す
data_dir = './hymenoptera_data'
import os
test_dir = os.path.join(data_dir, 'val')

# 画像
test_data = datasets.ImageFolder(test_dir, 
            transform=test_transform)

# バッチサイズ定義
batch_size = 50

# テスト画像
test_loader = DataLoader(
    test_data, batch_size = batch_size, 
    shuffle = False)

classes = ['ants', 'bees']

from torchvision import models
#ResNet50を選択
net = models.resnet50(pretrained = True)

# 最終レイヤー関数の入力次元数を確認
fc_in_features = net.fc.in_features

# 最終レイヤー関数の付け替え
n_output = len(classes)
net.fc = nn.Linear(fc_in_features, n_output)

# デバイス = cpuに設定
net = net.to(device)

# モデルを読み込む(ロードする)
net.load_state_dict(torch.load('net.prm'))

# 推論機能にする
net.eval() 

# 50個の写真の正誤
show_images_labels(test_loader, classes, net, device)


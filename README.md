# Recognizer_ants-bees
pytorchでハチと蟻の識別器を作成 またLocal PCで推論のみを行った

recognizer_ants-bees.jpynb
------------------------------

- ハチと蟻の画像の識別器の作成、及び推論結果の出力を行う
- 画像群はhttps://download.pytorch.org/tutorial/hymenoptera_data.zipから用いた
- google colaboratory　上で実行することを想定
- 上から実行すれば問題は無い


net.prm
------------------------------

- recognizer_ants-beesで作成した識別器のパラメータデータ


pusher_ants-bees.py
------------------------------

- net.prmを用いてlocal PC上で推論のみを行う為のプログラム
- python pusher_ants-bees.pyで実行

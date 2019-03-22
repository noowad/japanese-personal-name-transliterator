# 綴（深層学習ベースの外国人名カタカナ自動推定システム）

## 構成
- log: すべての人名データを学習データに利用して事前学習したモデルディレクトリ
- datas: 語彙定義・カタカナの正規化に必要なファイルをまとめたディレクトリ
- data_load.py: データのロードを担当
- graph.py: 学習モデルを定義
- hyperparams.py: ハイパーパラメータ
- tsuduri.py: API公開用
- test.py: テストデータに対する人名を推定
- module.py: モデルの構成要素(モジュール)
- network.py: モジュールを用いて構成したnetwork
- params.py: 学習するパラメータ
- rnn_wrappers.py: rnnをラッピング
- train.py: モデルを学習
- utils.py: その他のメソッド
- calc_performance.py: テストデータに対する性能を評価

## データ
- 各国の学習データを ./datas/nations/<IOC_code>.txtの形式にする。Country-unknownデータはUNK.txtにする。
- 基本的に語彙ファイル(eng_voca.txt, jap_voca.txt, nations.txt)はデフォルトの設定で良い。
- 人名データファイルの各行は以下のような形式で定義する。
- <source_name>+"\t"+<target_name>+"\n"

## 学習方法
1. まず、hyper-parameterを定義する。基本的に引数として渡すようになっているが、今のところ、デフォルト設定がもっともいい性能を出せる。
2. python train.pyでモデルを学習する。モデルはデフォルト設定では、./log/tacotron-basedディレクトリにセーブされる。
(CPU (3.7GHz Quad-Core Intel Xeon E5)では約3~4時間かかる。)
3. python test.pyでモデルの性能を評価する。

## 事前学習モデル
- 事前学習モデルは現在./log/tacotron_basedに含まれている。
- 国籍なしデータ約20万件および、202カ国の国籍付きデータ（国別推定ができる国はデータの数50個以上であった89カ国）約3万件の人名データを用いて学習したモデル。
## 事前学習したモデルによる推定方法
>>> from tsuduri import Tsuduri

>>> transliterator= Tsuduri()

>>> transliterator.to_katakana(name='Michael Jackson', country='USA', top_k=5)

['マイケル・ジョーダン', 'ミハエル・ヨルダン', 'ミカエル・ジョルダン', 'ミシェル・ホルダン', 'ミヒャエル・ヨーダン']

## Notes
- 本システムを商用目的で使用することはできません。
- Python3.6.3, Tensorflow 1.13.1の環境で正常に使用できることを確認 (Warningを消したいなら、Tensorflow バージョンを1.12.0以下に下げること)。
- おそらくTensorflow 2.0以上では正常に利用できない。
- Python2.xではなく、Python3.xを使うこと。(Python2.xではstr文字列とunicodeが統一されてないなど文字列の処理がややこしい。おそらく上手く利用できないと思う。)
- 現在はTacotron-basedモデル構造を採用しているが、今後Transformer-basedモデルを追加する予定である。

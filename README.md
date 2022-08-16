# AHC013（日本橋ハーフマラソン2022夏）
グリッド上でクラスタを作る問題。

# 必要そうなもの
- 盤面に対して連結成分を計算して、連結成分を大きさの降順でソートしておいて持つ

# 考察メモ

## 8/9
正の得点を得る。ランダムに動かして連結成分を計算。大きさが2以上のところを接続。

## 8/10
動かさずに接続可能なコンピュータをすべて接続すると26263点。これを初期解にして、動かすのを頑張るゲームな気がしてきた。
順位表では26万点くらいが1位なので、1ケースあたり5000点を取りたい。

できるだけ大きいクラスタを作るのが大事。以下はBinom(x, 2)である。

```bash
2 : 1
3 : 3
4 : 6
5 : 10
6 : 15
7 : 21
8 : 28
9 : 36
10 : 45
11 : 55
12 : 66
13 : 78
14 : 91
15 : 105
16 : 120
17 : 136
18 : 153
19 : 171
20 : 190
21 : 210
22 : 231
23 : 253
24 : 276
25 : 300
26 : 325
27 : 351
28 : 378
29 : 406
30 : 435
31 : 465
32 : 496
33 : 528
34 : 561
35 : 595
36 : 630
37 : 666
38 : 703
39 : 741
40 : 780
41 : 820
42 : 861
43 : 903
44 : 946
45 : 990
46 : 1035
47 : 1081
48 : 1128
49 : 1176
50 : 1225
51 : 1275
52 : 1326
53 : 1378
54 : 1431
55 : 1485
56 : 1540
57 : 1596
58 : 1653
59 : 1711
60 : 1770
61 : 1830
62 : 1891
63 : 1953
64 : 2016
65 : 2080
66 : 2145
67 : 2211
68 : 2278
69 : 2346
70 : 2415
71 : 2485
72 : 2556
73 : 2628
74 : 2701
75 : 2775
76 : 2850
77 : 2926
78 : 3003
79 : 3081
80 : 3160
81 : 3240
82 : 3321
83 : 3403
84 : 3486
85 : 3570
86 : 3655
87 : 3741
88 : 3828
89 : 3916
90 : 4005
91 : 4095
92 : 4186
93 : 4278
94 : 4371
95 : 4465
96 : 4560
97 : 4656
98 : 4753
99 : 4851
100 : 4950
```

大きさ77くらいのクラスタと大きさ64くらいのクラスタを1つずつ作れれば5000点。

scoreがseed:0, score=5052のとき、
```
3004,3289,3283,3289,3301,3301,3301,3293,3551,3603,3657,3665,3657,3596,4056,4092,4078,4092,
4092,4092,3915,3972,4031,5052
```
のように変わる。うーん。焼きなましにくそう？

# 8/11
コンピュータを動かす順序を焼きなまし？
近傍は挿入、削除、複製（続けて動かすと強そうなので）？

どこに動かすかは探索かな

15%くらい移動、85%くらい接続に操作回数を使いたい

# 8/12
最初のgridをcellに変換するとき、
```Rust
Cell::Computer { index: cs.len() };
```
が正しく、以前の実装ではコンピュータの種類をindexにしていて間違っていた。
でもまたスコア計算関数がバグっている。

↑直した。操作回数の上限を気にせずにスコア計算していた。

greedyとかもう使わないので整理していく。

改善したいところ
- movesの改善
    - ランダムに挿入や削除をしているが、もう少し意味のある挿入や削除をしたい
        - 連続で動かすような挿入など、狙いを持たせる
        - 何手か探索する
        - moveableから選ばれるcomputer.kindをroulette wheel selection
    - duplicate近傍など近傍を増やしたい
    - いちいち最初からシミュレーションしてるが差分更新したい
- connectsの改善
    - 閉路除去したい
    - 連携成分のサイズが大きいところから（大きくなるように）繋ぎたい
    - スコア計算と密な実装だが、切り分けてたい。なんらかの移動に対して柔軟に接続を切れるようにしたい。
- /tools下に並列にテスト実行するコードを書きたい `cargo run -p tools --bin test`みたいなのができるやつ
- 0003.txtが2秒で2011、20秒で4637なので高速化すると嬉しい

- いちいち最初からシミュレーションしてるが差分更新したい
これから実装する。
まずinsertの候補となる`moveable`を差分更新で済むようにしたい。
これは`Computer.go()`で動かしたときと、ケーブルをつないだ時に変化する。現状ケーブルは動かし終わってから繋ぐので無視してよい？

insertは途中でやりたいので、`moveable`は`new_moves.len()`個必要。うーんmoveableそのまま持ってなくても、移動可能なcomputerの集合を持っていればよさそう。毎回`100*k`個のcomputerを調べるのが嫌？`移動回数*100*k`個の移動可能コンピュータ集合を持っておく？でもこれを保持するようなやつを作ろうとするとどうせ`100*k`ステップかかるな…

gridとcomputersは差分更新できる？

gridをcloneするより、movesがprevとnextを持っておいて、insertのときとかは適宜戻す方が早い？
insertなどが失敗したときはmovesによって復元できるようにする。うーん、実装が複雑すぎる気がする。

gridは最新からcloneしてremoveやinsertするところまで戻すためにprevを持つ。これはありだけど今やらなくてもいい気がしてきた。

意外とよくかけているな…

insertの上限を40%くらいにしてたけど、こういうのは評価関数に任せた方がいいんだよな　撤廃

- スコア計算と密な実装だが、切り分けてたい。なんらかの移動に対して柔軟に接続を切れるようにしたい。
次はここらへんを見ていきたい

接続をどうやって持っておこう…。
各コンピュータがどのコンピュータにつながるか、またその方向を持つ
スコア計算関数にて、ufは計算する？（まだ連結成分ごとのデータを保持していないので）
繋ぐときは両方の隣接リストに書き込み　消すときも両方の隣接リストから削除（どうせ次数は最大4）

# 8/13

焼きなまし前の初期解削除で伸びた。変に固定するの、よくない。

- スコア計算と密な実装だが、切り分けたい。なんらかの移動に対して柔軟に接続を切れるようにしたい
難しい。
つなぐ順番も焼きなます？

スコア計算の実装が複雑になっていた部分を丁寧に切り分けることで高速化し、160kまできた。

0007.txtのようなk=5でほぼ空白がない、みたいなやつはどうしたら上がるんだ？
接続も焼きなました方がいい気がしてきた してきたが実装がつらい。

cableを消すところ、Computer.goに含めたい
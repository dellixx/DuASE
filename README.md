<h2 align="center">
Learning Low-dimensional Multi-domain Knowledge Graph Embedding via Dual Archimedean Spirals
</h2>




## 🔬 Dependencies
- Python >= 3.6
- PyTorch >= 1.0




## 📜 Statistics of mutil-domain $n$-MDKG


| Dataset |  $E$   |  $R$  | #Train  | #Valid | #Test |
|---------|--------|-------|---------|--------|-------|
| 3-MDKG   |  8,691 |   69  |  90,130 |   918  |  922  |
| 6-MDKG   | 90,275 |  361  | 429,810 | 4,383  | 4,391 |
| 9-MDKG   |102,880 |  426  | 456,281 | 5,200  | 5,211 |


+ 3-MDKG dataset covers 3 domains including education, film, and sports.

+ 6-MDKG dataset covers 6 domains including medicine, education, film, sports, politics, and dictionary.

+ 9-MDKG dataset covers 9 domains including medicine, education, film, sports, politics, dictionary, geography, automotive, and modern stars.

## 🚀 Reproduce the Results on multi-domain ($n$-MDKG) KGs





### 3MDKG



```console
python run.py --data_path data/3mdkg --dataset 3mdkg --do_train --cuda --do_valid --do_test --model DuASE -n 512 -b 1024 -d 32 -g 6 -a 1 -adv -de -tcr -lr 0.001 --max_steps 60000 --cpu_num 8 --test_batch_size 16 --regularization 0.25 -randomSeed 4
```



### 6MDKG



```console
python run.py --data_path data/6mdkg --dataset 6mdkg --do_train --cuda --do_valid --do_test --model DuASE -n 512 -b 1024 -d 32 -g 11 -a 1 -adv -de -tcr -lr 0.005 --max_steps 60000 --cpu_num 8 --test_batch_size 16 --regularization 0.25 -randomSeed 4
```

### 9MDKG



```console
python run.py --data_path data/9mdkg --dataset 9mdkg --do_train --cuda --do_valid --do_test --model DuASE -n 512 -b 1024 -d 32 -g 8 -a 1 -adv -de -tcr -lr 0.001 --max_steps 60000 --cpu_num 8 --test_batch_size 16 --regularization 0.2 -randomSeed 4
```

## 🚀 Reproduce the Results on "single"-domain KGs

### WN18RR
```console
python run.py --data_path data/wn18rr --dataset wn18rr --do_train --cuda --do_valid --do_test --model DuASE -n 512 -b 1024 -d 32 -g 6 -a 1 -adv -de -tcr -n2 -lr 0.001 --max_steps 80000 --cpu_num 8 --test_batch_size 8 --regularization 0.4 -randomSeed 4
```


### FB15K-237
```console
python run.py --data_path data/fb15k-237 --dataset fb15k-237 --do_train --cuda --do_valid --do_test --model DuASE -n 512 -b 2048 -d 32 -g 8 -a 0.5 -adv -de -tcr -n2 -lr 0.005 --max_steps 80000 --cpu_num 8 --test_batch_size 32 --regularization 0.2 -randomSeed 4
```



### YAGO3-10
```console
python run.py --data_path data/yago3-10 --dataset yago3-10 --do_train --cuda --do_valid --do_test --model DuASE -n 512 -b 1024 -d 32 -g 24 -a 1 -adv -de -tcr -lr 0.005 --max_steps 80000 --cpu_num 8 --test_batch_size 8 --regularization 0.4 -randomSeed 4
```

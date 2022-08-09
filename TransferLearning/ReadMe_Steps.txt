
Step 1:
git clone https://github.com/bert-nmt/bert-nmt.git

download training and testing dataset en, fr, cs and de from https://github.com/multi30k/dataset/tree/master/data/task1/raw

Step 2:

copy files "prepare-multilingual.sh" to the path '../bert-nmt/examples/translation/'

execute the script ../bert-nmt/examples/translation/prepare-multilingual.sh


step 3:

execute the following script to preprocess data

TEXT=examples/translation/multi30k
DESTDIR=examples/translation/destdir_fr
fairseq-preprocess --source-lang fr --target-lang en \
  --trainpref $TEXT/train.bpe --testpref $TEXT/test.bpe \
  --destdir $DESTDIR 
  
TEXT=examples/translation/multi30k
DESTDIR=examples/translation/destdir_cs
fairseq-preprocess --source-lang cs --target-lang en \
  --trainpref $TEXT/train.bpe --testpref $TEXT/test.bpe \
  --destdir $DESTDIR 
  
TEXT=examples/translation/multi30k
DESTDIR=examples/translation/destdir_de
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train.bpe --testpref $TEXT/test.bpe \
  --destdir $DESTDIR 


Step 4:

execute the following script to train de2en, cs2en and fr2en

src=de
tgt=en
bedropout=0.5
ARCH=transformer
DATAPATH=../bert-nmt/examples/translation/destdir_de
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}
mkdir -p $SAVEDIR
if [ ! -f $SAVEDIR/checkpoint_nmt.pt ]
then
    cp /your_pretrained_nmt_model $SAVEDIR/checkpoint_nmt.pt
fi
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler"
else
warmup=""
fi

python train.py $DATAPATH \
-a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
--max-epoch 300 \
--dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 150000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --warmup-from-nmt --reset-lr-scheduler \
--encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout | tee -a $SAVEDIR/training.log\
--bert-model-name bert-base-multilingual-uncased



src=cs
tgt=en
bedropout=0.5
ARCH=transformer
DATAPATH=../bert-nmt/examples/translation/destdir_cs
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}
mkdir -p $SAVEDIR
if [ ! -f $SAVEDIR/checkpoint_nmt.pt ]
then
    cp /your_pretrained_nmt_model $SAVEDIR/checkpoint_nmt.pt
fi
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler"
else
warmup=""
fi

python train.py $DATAPATH \
-a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
--max-epoch 300 \
--dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 150000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --warmup-from-nmt --reset-lr-scheduler \
--encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout | tee -a $SAVEDIR/training.log\
--bert-model-name bert-base-multilingual-uncased



src=fr
tgt=en
bedropout=0.5
ARCH=transformer
DATAPATH=../bert-nmt/examples/translation/destdir_fr
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}
mkdir -p $SAVEDIR
if [ ! -f $SAVEDIR/checkpoint_nmt.pt ]
then
    cp /your_pretrained_nmt_model $SAVEDIR/checkpoint_nmt.pt
fi
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler"
else
warmup=""
fi

python train.py $DATAPATH \
-a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
--max-epoch 300 \
--dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 150000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --warmup-from-nmt --reset-lr-scheduler \
--encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout | tee -a $SAVEDIR/training.log\
--bert-model-name bert-base-multilingual-uncased


Step 5:

execute the following script to output fr2fr 

src=fr
tgt=en
ARCH=checkpoints/iwed_fr_en_0.5/checkpoint300.pt
DATAPATH=examples/translation/destdir_fr

python generate_fr.py $DATAPATH \
--path $ARCH -s $src -t $tgt --label-smoothing 0.1 \
--criterion label_smoothed_cross_entropy \
--bert-model-name bert-base-multilingual-uncased \
--gen-subset train



execute the following script to output de2fr 

src=de
tgt=en
ARCH=checkpoints/iwed_de_en_0.5/checkpoint300.pt
DATAPATH=examples/translation/destdir_de

python generate_fr.py $DATAPATH \
--path $ARCH -s $src -t $tgt --label-smoothing 0.1 \
--criterion label_smoothed_cross_entropy \
--bert-model-name bert-base-multilingual-uncased \
--gen-subset train



execute the following script to output cs2fr 

src=cs
tgt=en
ARCH=checkpoints/iwed_cs_en_0.5/checkpoint300.pt
DATAPATH=examples/translation/destdir_cs

python generate_fr.py $DATAPATH \
--path $ARCH -s $src -t $tgt --label-smoothing 0.1 \
--criterion label_smoothed_cross_entropy \
--bert-model-name bert-base-multilingual-uncased \
--gen-subset train



generate_cs.py and generate_de.py are scripts for decoders of cs and de.
printf "downloading LjSpeech...\n"
axel -n 8 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -q
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1

gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt data/

printf "downloading waveglow...\n"
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt

printf "downloading mels...\n"
gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz
echo $(ls mels | wc -l)
mv mels data/

printf "downloading alignments...\n"
axel -n 8 https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip >> /dev/null
mv alignments data/

git clone https://github.com/xcmyz/FastSpeech.git
mv FastSpeech/waveglow/* .
mv FastSpeech/glow.py .

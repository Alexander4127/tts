printf "downloading waveglow...\n"
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt

git clone https://github.com/xcmyz/FastSpeech.git
mv FastSpeech/waveglow/* waveglow/
mv FastSpeech/glow.py .

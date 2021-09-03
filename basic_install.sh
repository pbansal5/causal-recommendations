cp -r /ecstorage/bert-opt/src .
sudo-apt get install tmux
cd src
pip3 install transformers
pip3 install sentence_transformers
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
python3 setup.py install --user
cd ..
pip3 install nlpaug
pip3 install gpustat
sudo apt-get install emacs
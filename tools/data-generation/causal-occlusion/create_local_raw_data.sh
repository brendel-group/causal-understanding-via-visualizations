echo "install packages"

echo "sudo pip install --upgrade pip"
sudo pip install -q --upgrade pip
echo "sudo pip3 install lucid==0.3.8"
sudo pip3 install -q lucid==0.3.8
echo "sudo apt update"
sudo apt update
echo "sudo apt install -y libgl1-mesa-glx"
sudo apt install -y libgl1-mesa-glx
sudo "sudo pip install opencv-python"
sudo pip install -q opencv-python
echo "sudo pip install jupytext"
sudo pip install -q jupytext

echo "nb 00_generate_layer_folder_mapping_csv_for_all_feature_maps.ipynb"
jupytext --to py 00_generate_layer_folder_mapping_csv_for_all_feature_maps.ipynb
echo "nb 00"
python3 00_generate_layer_folder_mapping_csv_for_all_feature_maps.py

echo "nb 01_natural_stimuli_activations.ipynb"
jupytext --to py 01_natural_stimuli_activations.ipynb
echo "nb 01"
python3 01_natural_stimuli_activations.py

echo "Cleaning up"
rm 00_generate_layer_folder_mapping_csv_for_all_feature_maps.py
rm 01_natural_stimuli_activations.py
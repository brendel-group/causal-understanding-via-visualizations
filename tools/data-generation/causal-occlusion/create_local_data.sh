echo "Usage: create_local_data.sh DATA_FOLDER TRIAL_TYPE [install]"

if [ $3 = "install" ]; then
  echo "install packages"

  echo "sudo pip install --upgrade pip"
  sudo sudo `which pip` install -q --upgrade pip
  echo "sudo pip3 install lucid==0.3.8"
  sudo sudo `which pip` install -q lucid==0.3.8
  echo "sudo apt update"
  sudo apt update
  echo "sudo apt install -y libgl1-mesa-glx"
  sudo apt install -y libgl1-mesa-glx
  sudo "sudo pip install opencv-python"
  sudo sudo `which pip` install -q opencv-python
  echo "sudo pip install jupytext"
  sudo sudo `which pip` install -q jupytext
  echo "sudo pip install pandas"
  sudo sudo `which pip` install -q pandas
else
  echo "Not installing packages since 3rd argument was not set to 'install'"
fi


if [ -z ${1+x} ]; then
  echo "First argument (data folder path) is unset";
  exit -1;
else
  data="$1"
fi

set -e

echo "nb 1_save_natural_reference_and_default_images.ipynb"
jupytext --to py 1_save_natural_reference_and_default_images.ipynb
echo "nb 1 $2"
python3 1_save_natural_reference_and_default_images.py -s=$DATAPATH/${data} -t=$2

echo "nb 2_occlusion_activations_in_Inception_V1.ipynb"
jupytext --to py 2_occlusion_activations_in_Inception_V1.ipynb
echo "nb 2 $2"
python3 2_occlusion_activations_in_Inception_V1.py -s=$DATAPATH/${data} -t=$2

echo "nb 3_occlusion_save_query_images.ipynb"
jupytext --to py 3_occlusion_save_query_images.ipynb
echo "nb 3 $2"
python3 3_occlusion_save_query_images.py -s=$DATAPATH/${data} -t=$2

echo "nb 4_save_synthetic_reference_images.ipynb"
jupytext --to py 4_save_synthetic_reference_images.ipynb
echo "nb 4 $2"
python3 4_save_synthetic_reference_images.py -s=$DATAPATH/${data} -t=$2

echo "nb 8_blur_activations_and_save_maximally_activating_blur_img.ipynb"
jupytext --to py 8_blur_activations_and_save_maximally_activating_blur_img.ipynb
echo "nb 8 $2"
if [[ $data == *pure* ]]; then
  python3 8_blur_activations_and_save_maximally_activating_blur_img.py -s=$DATAPATH/${data} -t=$2
else;
 echo "Skipping blurred images"
fi

echo "Cleaning up"
rm 1_save_natural_reference_and_default_images.py
rm 2_occlusion_activations_in_Inception_V1.py
rm 3_occlusion_save_query_images.py
rm 4_save_synthetic_reference_images.py
rm 8_blur_activations_and_save_maximally_activating_blur_img.py
if [ -z ${1+x} ]; then
  echo "First argument (data folder path) is unset";
  exit -1;
else
  data="$1"
fi

if [ -z ${2+x} ]; then
  echo "Second argument (raw data folder path) is unset";
  exit -1;
else
  raw_data="$2"
fi

if [ -d "${data}" ]; then
  echo "Data folder '${data}' exists."
  while true; do
    read -p "Do you wish to clear the data folder [y/n]? " yn
    case $yn in
        [Yy]* ) echo "Clearing folder '${data}'..."; rm -r "${data}"; mkdir -p "${data}"; break;;
        [Nn]* ) echo "WARNING: Keeping the existing data folder; this can result in unused/stale files."; break;;
        * ) echo "Please answer yes (y) or no (n).";;
    esac
done
else
  mkdir -p "${data}"
fi

echo "Generating structure"
python3 create_task_structure_json.py -nt=18 -nc=3 -nh=260 -k 1 3 -c 40 -s=${raw_data}/stimuli_pure_conditions/channel -o natural.json -m=natural --seed=1
python3 create_task_structure_json.py -nt=18 -nc=3 -nh=260 -k 1 3 -c 40 -s=${raw_data}/stimuli_pure_conditions/channel -o optimized.json -m=optimized --seed=1
python3 create_task_structure_json.py -nt=18 -nc=3 -nh=260 -k 1 3 -c 40 -s=${raw_data}/stimuli_mixed_conditions/channel -o mixed.json -m=mixed --seed=1
python3 create_task_structure_json.py -nt=18 -nc=3 -nh=260 -k 1 3 -c 40 -s=${raw_data}/stimuli_pure_conditions/channel -o natural_blur.json -m=natural-blur --seed=1

echo
echo "Adding baselines to structure"
python add_relative_activation_difference_baseline_values_to_structure.py --input-structure=natural.json --output-structure=natural_with_baselines.json --occlusion-size=40
python add_relative_activation_difference_baseline_values_to_structure.py --input-structure=optimized.json --output-structure=optimized_with_baselines.json --occlusion-size=40
python add_relative_activation_difference_baseline_values_to_structure.py --input-structure=mixed.json --output-structure=mixed_with_baselines.json --occlusion-size=40
python add_relative_activation_difference_baseline_values_to_structure.py --input-structure=natural_blur.json --output-structure=natural_blur_with_baselines.json --occlusion-size=40

python add_center_baseline_values_to_structure.py --input-structure=natural_with_baselines.json --output-structure=natural_with_baselines.json --occlusion-size=40
python add_center_baseline_values_to_structure.py --input-structure=optimized_with_baselines.json --output-structure=optimized_with_baselines.json --occlusion-size=40
python add_center_baseline_values_to_structure.py --input-structure=mixed_with_baselines.json --output-structure=mixed_with_baselines.json --occlusion-size=40
python add_center_baseline_values_to_structure.py --input-structure=natural_blur_with_baselines.json --output-structure=natural_blur_with_baselines.json --occlusion-size=40

python add_variance_baseline_values_to_structure.py --input-structure=natural_with_baselines.json --output-structure=natural_with_baselines.json --occlusion-size=40
python add_variance_baseline_values_to_structure.py --input-structure=optimized_with_baselines.json --output-structure=optimized_with_baselines.json --occlusion-size=40
python add_variance_baseline_values_to_structure.py --input-structure=mixed_with_baselines.json --output-structure=mixed_with_baselines.json --occlusion-size=40
python add_variance_baseline_values_to_structure.py --input-structure=natural_blur_with_baselines.json --output-structure=natural_blur_with_baselines.json --occlusion-size=40

echo "Installing dependencies (if necessary)"
sudo `which pip` install boltons pysaliency glom
echo "Downloading deepgaze model weights"
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/YW8LRMJiboKnbXG/download -O deepgaze_pytorch/DeepGazeII_DSREx3.pth
python add_saliency_baseline_values_to_structure.py --input-structure=natural_with_baselines.json --output-structure=natural_with_baselines.json --occlusion-size=40
python add_saliency_baseline_values_to_structure.py --input-structure=optimized_with_baselines.json --output-structure=optimized_with_baselines.json --occlusion-size=40
python add_saliency_baseline_values_to_structure.py --input-structure=mixed_with_baselines.json --output-structure=mixed_with_baselines.json --occlusion-size=40
python add_saliency_baseline_values_to_structure.py --input-structure=natural_blur_with_baselines.json --output-structure=natural_blur_with_baselines.json --occlusion-size=40


echo
echo "Copying stimuli to populate structure"
python3 create_task_structure_from_json.py -t ${data}/no_references -i natural.json -nr 0
python3 create_task_structure_from_json.py -t ${data}/natural_9_references -i natural.json -nr 9
python3 create_task_structure_from_json.py -t ${data}/optimized_9_references -i optimized.json -nr 9
python3 create_task_structure_from_json.py -t ${data}/natural_5_optimized_4_references -i mixed.json -nr 9
python3 create_task_structure_from_json.py -t ${data}/natural_blur_9_references -i natural_blur.json -nr 9

echo
echo "Copy json structures to output folders"
cp optimized.json ${data}/optimized_9_references/
cp natural.json ${data}/natural_9_references/
cp mixed.json ${data}/natural_5_optimized_4_references/
cp natural_blur.json ${data}/natural_blur_9_references/
cp natural.json ${data}/no_references/

cp optimized_with_baselines.json ${data}/optimized_9_references/
cp natural_with_baselines.json ${data}/natural_9_references/
cp mixed_with_baselines.json ${data}/natural_5_optimized_4_references/
cp natural_blur_with_baselines.json ${data}/natural_blur_9_references/
cp natural_with_baselines.json ${data}/no_references/

echo
echo "Copy instruction screenshots"
cp -r instructions/natural ${data}/natural_9_references/instructions
cp -r instructions/optimized ${data}/optimized_9_references/instructions
cp -r instructions/none ${data}/no_references/instructions
cp -r instructions/natural_blur ${data}/natural_blur_9_references/instructions
cp -r instructions/natural_optimized ${data}/natural_5_optimized_4_references/instructions

echo
echo "Cleaning up"
rm natural.json
rm optimized.json
rm mixed.json
rm natural_blur.json
rm natural_with_baselines.json
rm optimized_with_baselines.json
rm mixed_with_baselines.json
rm natural_blur_with_baselines.json

if [ -z ${1+x} ]; then
  echo "First argument (data folder path) is unset";
  exit -1;
else
  data="$1"
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

# Replace with path to data generated with code form https://github.com/bethgelab/testing_visualizations
$PATHTORAWSTIMULI="TODO"

echo "Generating structure"
python create_task_structure_json.py -nt=9 -nc=3 -nh=130 -k=1 -s=$PATHTORAWSTIMULI/channel/sampled_trials/ -o natural.json -m=natural --seed=1
python create_task_structure_json.py -nt=9 -nc=3 -nh=130 -k=1 -s=$PATHTORAWSTIMULI/channel/sampled_trials/ -o optimized.json -m=optimized --seed=1

echo "Copying stimuli to populate structure"
python create_task_structure_from_json.py -t ${data}/optimized_9_references -i optimized.json -nr 9
python create_task_structure_from_json.py -t ${data}/natural_9_references -i natural.json -nr 9

echo "Copy json structures to output folders"
cp optimized.json ${data}/optimized_9_references/
cp natural.json ${data}/natural_9_references/

echo "Copy instruction screenshots"
cp -r instructions/natural_9_references ${data}/natural_9_references/instructions
cp -r instructions/optimized_9_references ${data}/optimized_9_references/instructions

echo "Cleaning up"
rm natural.json
rm optimized.json
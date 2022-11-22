search_dir=/ssd2/tangshiyu/Code/PaddleSeg/data/Mask_Iron_new/mask_iron/
i=0
dir=/ssd2/tangshiyu/Code/PaddleSeg/data/Mask_Iron_new/mask_iron/convert_annotation
for entry in "$search_dir"*.json
do
  echo "$entry"
  mkdir $dir$i
  labelme_json_to_dataset $entry  -o $dir$i
  i=`expr $i + 1`
  echo "$dir$i" 
done
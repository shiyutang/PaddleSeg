# 训练宝钢数据
本次基于宝钢labelme标注的数据使用pp-liteseg模型进行训练。

## 0. 搭建环境
```
unzip paddleseg.zip

conda activate paddle2.3.2  # 开启安装了paddle2.3.2的环境

cd PaddleSeg/
pip install -r requirements.txt
pip install labelme
```

## 1. labelme格式转换：
使用convert_labelme.sh进行数据转换，如有需要，修改其中的search_dir为你自己的数据路径，运行成功后会在search_dir生成以convert_annotationi为名字的文件夹，每个文件夹内部是一个数据转换后的结果，包含图像和标签。（自己的数据需要下载最新的钢筋数据）

```bash
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
```

## 2. 数据集划分：
将转化好的数据按照大概80/15/5的比例划分为训练集、评估集和测试集，并放入到`data/Mask_Iron_new/dataset`的目录下，目录结构如下所示：
```
data/Mask_Iron_new/dataset
    ├── test
    ├── train
    └── val
```

## 3. 难例数据重采样扩充：
由于训练数据存在难易不均衡的问题，所以需要手动扩充较难的数据，例如`202206150712219239604.jpg`就是一个较难的数据，他具有钢筋不对齐的问题。
通过手动数据重采样，我们需要达到较难数据和简单数据为1:1的情况。

## 4. 补充模糊和垂直翻转数据增广进行训练：
数据增广已经在`configs/iron/pp_liteseg.yml`中实现，在训练中加载即可，如有需要，可以使用`vim configs/iron/pp_liteseg.yml` 修改。

```python
python3  train.py  --config configs/iron/pp_liteseg.yml --use_vdl  --save_dir output/mask_iron --save_interval 1000 --log_iters 300  --num_workers 4 --do_eval  
--keep_checkpoint_max 10

visualdl --log_dir output/mask_iron # 查看训练效果

python -m paddle.distributed.launch val.py --config configs/iron/pp_liteseg.yml  --model_path output/mask_iron/best_model/model.pdparams  # 动态图验证效果
```

## 5. 导出到静态图并使用部署：
```bash
# 导出
python export.py  --config configs/iron/pp_liteseg.yml  --models output/mask_iron/best_model/model.pdparams --save_dir output/mask_iron/export

# 下载图片
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# 部署
python deploy/python/infer.py     --config output/mask_iron/export/deploy.yaml     --image_path ./cityscapes_demo.png  --use_trt True
```

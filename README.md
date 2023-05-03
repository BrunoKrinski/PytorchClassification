<h1>Classification with Pytorch Networks</h1>

<h2>Installation: </h2>

<p>Install Python 3.10.10</p>
<p>Install CUDA Toolkit 11.8</p>
<p>Install Pytorch 2.0</p>
<p>Install Other Requeriments: pip install -r requeriments.txt</p>

<h2>Organized Your Dataset: </h2>

```
|-- Dataset_Name/
|   |-- train/
|   |   |-- label_1/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_2/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_3/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- ...
|   |-- test/
|   |   |-- label_1/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_2/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_3/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- ...
|   |-- valid/
|   |   |-- label_1/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_2/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- label_3/
|   |   |   |-- image1.jpg
|   |   |   |-- image2.jpg
|   |   |   |-- image3.jpg
|   |   |   |-- ...
|   |   |-- ...
...
```

<h2>Execute: </h2>

<h3>Train: </h3>
<p>python trainer.py --dataset dataset_path --output output_folder --size images_size --epochs num_epochs --batch_size batch_size --model model_name --learning_rate learning_rate</p>

<h3>Test: </h3>
<p>python classify.py --model path_to_model.pt --image image_path --size image_size --dataset dataset_path</p>

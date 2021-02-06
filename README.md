<br><br><br>
-->
# CycleGAN

An implementation of CycleGan for learning an image-to-image translation without input-output pairs using Keras.

Original paper: https://arxiv.org/abs/1703.10593

You can check architecture of our models here: [Generator](images/generator_model_plot.png), [Discriminator](images/discriminator_model_plot.png)

<img src="images/model.jpg">

# Results

### - Monet -> Photo <br>
<img src="images/monet2photo/227_0.png" width="300px"/> <img src="images/monet2photo/254_0.png" width="300px"/> <img src="images/monet2photo/269_0.png" width="300px"/> <img src="images/monet2photo/280_0.png" width="300px"/> <img src="images/monet2photo/288_0.png" width="300px"/> <img src="images/monet2photo/292_0.png" width="300px"/> <img src="images/monet2photo/294_0.png" width="300px"/> <img src="images/monet2photo/296_0.png" width="300px"/> <img src="images/monet2photo/297_0.png" width="300px"/>

### - Photo -> Portrait <br>


## Data preparing
 * Dounload a dataset, e.g portrait2photo
 ```bash
 $ ./bin/datasets.sh portrait2photo
 ```

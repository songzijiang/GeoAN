# GeoAN
The official code of [MDG625: a daily high-resolution meteorological dataset derived by geopotential-guided attention network in Asia (1940-2023)"](https://essd.copernicus.org/articles/17/1501/2025/essd-17-1501-2025-discussion.html).

Any questions can be touched by jacksung1995@gmail.com
## Data download

1. The low resolution data ERA5 can be downloaded from [ECMWF](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).

2. The downloaded data needs to be organized into the appropriate format. The order of the parameters is ['WIN', 'TMP', 'PRS', 'PRE'].
[Here are some examples in 2023-12](https://drive.google.com/file/d/1ExjsISNm1bWdUimhNEhu6duJRq9ejsoW/view?usp=drive_link).

3. The pre-trained model can be downloaded from [here](https://drive.google.com/file/d/1OVrGFcdHiZkKUFcyWm28QQAE8iDbA-Pk/view?usp=sharing).
## Code run
The configuration can be modified in the `configs/config.yml`.

The file struct should organized as:
```
|--data
    |--exclude
       |--2023
           |--12
               |--1
                  |--era5.np.npy
               |--2
                  |--era5.np.npy
               ...
           ...
       |--era5_global_std_level.npy
       |--era5_global_mean_level.npy
       |--cldas_global_std_level.npy
       |--cldas_global_mean_level.npy
       |--geoan.pt
```
Execute `sh test.sh` can run the test program to produce high-resolution results.

The results will be shown in the `proc` folder if each step is right.

The DOI of the produced dataset MDG625 is [10.57760/sciencedb.17408](https://doi.org/10.57760/sciencedb.17408).

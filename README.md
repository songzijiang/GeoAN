# GeoAN
The official code of [MDG625: A daily high-resolution meteorological dataset derived by geopotential-guided attention network in Asia (1940-2023)"]().

Any questions can be touched by jacksung1995@gmail.com
## Data download

1. The low resolution data ERA5 can be downloaded from [ECMWF](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).

2. The downloaded data needs to be organized into the appropriate format. The order of the parameters is ['WIN', 'TMP', 'PRS', 'PRE'].
[Here are some examples in 2023-12](https://drive.google.com/file/d/1ExjsISNm1bWdUimhNEhu6duJRq9ejsoW/view?usp=drive_link).

4. The mean and std data of the dataset can be downloaded by the following links:
[era5_global_std_level](https://drive.google.com/file/d/1V-xV6QbjRalvtVsA04A6LOXWmvhbuUDR/view?usp=drive_link), 
[era5_global_mean_level](https://drive.google.com/file/d/1qQQcfz9RQyW_KcKq_PH9_uWDfOz-mKMl/view?usp=drive_link), 
[cldas_global_std_level](https://drive.google.com/file/d/1lCGvOzFT00DyVekyq4L4N-8Ymx_RqqFP/view?usp=drive_link), 
[cldas_global_mean_level](https://drive.google.com/file/d/1QUxpgzHc14S4zbEORBRYeJin3pWZ-Mxy/view?usp=drive_link).
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

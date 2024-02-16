# GeoAN
The offical code of GeoAN (Downscaling Using Geopotential-Guided Attention Network for Historical Daily Meteorological Data Since 1940).

The first draft of the paper has been submitted to [GRL](https://agupubs.onlinelibrary.wiley.com/journal/19448007).

More details will be released when the papaer is in public, any question can be touch in jacksung@gmail.com
## Data download
1. Pre-trained model can be downloaded in [Google Drive](https://drive.google.com/file/d/1BN7EE0uVov2b4sTpSP8EKlXIxE6lDetI/view?usp=drive_link).

2. The low resolution data ERA5 can be downloaded from [ECMWF](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).

3. The downloaded data needs to be organized into the appropriate format. The order of the parameters is ['WIN', 'TMP', 'PRS', 'PRE'].
[Here are some examples in 2023-12](https://drive.google.com/file/d/1ExjsISNm1bWdUimhNEhu6duJRq9ejsoW/view?usp=drive_link).

4. The mean and std data of the dataset can be downloaded by the following links:
[era5_global_std_level](https://drive.google.com/file/d/1V-xV6QbjRalvtVsA04A6LOXWmvhbuUDR/view?usp=drive_link), 
[era5_global_mean_level](https://drive.google.com/file/d/1qQQcfz9RQyW_KcKq_PH9_uWDfOz-mKMl/view?usp=drive_link), 
[cldas_global_std_level](https://drive.google.com/file/d/1lCGvOzFT00DyVekyq4L4N-8Ymx_RqqFP/view?usp=drive_link), 
[cldas_global_mean_level](https://drive.google.com/file/d/1QUxpgzHc14S4zbEORBRYeJin3pWZ-Mxy/view?usp=drive_link).
## Code run
The configuation can be modified in the `configs/config.yml`.

The file struct should originzed as:
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
Excute `sh test.sh` can run the test program to produce the high-resolution results.

If each step is right, the results will be shown in the `proc` folder. The results of [2023-12](https://drive.google.com/file/d/1iH5lD84n5VORjXLfAlbQTGHSbrJbpT9T/view?usp=drive_link) can be downloaded.

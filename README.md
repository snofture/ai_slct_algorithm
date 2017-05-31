## AI Selection Project
### Requirment

run on linux os
hive, which can query bdp.jd.com
python, including numpy, pandas, sklearn, yaml
R, including datatable, bit64, yaml

### Tutorials

#### run a pipeline

python ai_slct_pipeline.148.py cid3 dt lvl
Here, cid3 is one or more level3 category id(s) joined by "-", eg. '1590' or '1595-5020'. dt is the last day of a month. eg. '2017-01-31'. lvl is 'lvl3' or 'lvl4', which means you take a level3 category as selection pool or take level4 category as selection pool.
So, if you want to select some good sku from baverages(cid3=1590) based on data of Feb 2017, and take each fourth category as a pool, you can run this:
python ai_slct_pipeline.148.py 1590 2017-02-28 lvl4
Sometimes, we want merge many level3 categorys into one pool. For example, Biscuit cake in JD include two level3 categorys: '1595' and '5020', you can do this:
python ai_slct_pipeline.148.py 1595-5020 2017-02-28 lvl3
If you want take each level4 category as a pool in above example, run this:
python ai_slct_pipeline.148.py 1595-5020 2017-02-28 lvl4
#### run a step of pipeline

Download data from bdp.jd.com my be very slow. If you have already downloaded all data for a level3 category, you can just run the computation part of pipeline
python computation_pipe.py cid3 lvl
at this time, you should not specify the dt parameter, as the data you need had been downloaded.
More specific modules can be run separately:
/software/servers/R-3.3.2-install/bin/Rscript switchR.R params/sku_1590.yaml

spark-submit --master  spark://172.19.142.130:7077 \
             --deploy-mode client  \
             --total-executor-cores 100 \
             --executor-memory 10G  \
             halo.py params/sku_1590.yaml

python sale_summary.py params/sku_1590.yaml

python cdt.py params/sku_1590.yaml

python switching_prediction.py params/sku_1590.yaml

python utility.py params/sku_1590.yaml

python brand_transform.py params/sku_1590.yaml

python sku_selection_new.py params/sku_1590.yaml
#### monitor the pipeline

Each time you run a level3 category, a log file will be created, take a look at it:
more logs/2017-01-31/1590.log
or monitor the computation progress:
tail -f logs/2017-01-31/1590.log
DO NOT delete this log until result have beed uploaded to bdp.jd.com, as the uploading step will search informations in this log
#### ignored attributes in cdt

any meaningless attribute will be ignored if you write it in here:
vim input/ignored_attrs.txt
#### user defined level4 category

Generally, for a level3 category, we automatically take different '分类' attributes as level4 categorys. You can also assign level4 category manually, try edit this file:
vim input/lvl4_desc.txt

# Experiments
## Table	5: Performance on TrDataSets
### TrMor2006
```shell
    sh trmor2006_full.sh
    sh trmor2016_full.sh
    sh trmor2018_full.sh
```

## Table 6: Mono-Lingual experiments

### 100sent
```shell
    sh low_resource_100sent_full.sh
    sh low_resource_100sent_tag.sh
```

### 1000sent
```shell
    sh low_resource_1000sent_full.sh
    sh low_resource_1000sent_tag.sh
```


## Table 7: Transfer-Learning experiments
### Source Language Training
```shell
    sh high_resource_da.sh
    sh high_resource_es.sh
    sh high_resource_fi.sh
    sh high_resource_ru.sh
```

### 100sent
```shell
    sh transfer_100sent_full.sh
    sh transfer_100sent_tag.sh
```
### 1000sent
```shell
    sh transfer_1000sent_full.sh
    sh transfer_1000sent_tag.sh
```

## Table 8: High-resource languages
Note: if you run below for Table 7, you don't need to run hem again.

```shell
    sh high_resource_da.sh
    sh high_resource_es.sh
    sh high_resource_fi.sh
    sh high_resource_ru.sh
```

## Table 9: Ablation experiments
Note: If you run some experiments in below for Table 5, you don't need to run hem again.

```shell
    sh trmor2018_full.sh
    sh trmor2018_full_context.sh
    sh trmor2018_full_s2s.sh
```

## Table 10: Decoder Experiments
Note: if you run below for Table 7, you don't need to run hem again.
### Seq
```shell
    sh high_resource_da.sh
    sh high_resource_es.sh
    sh high_resource_fi.sh
    sh high_resource_ru.sh
```
#### Tag
```shell
    sh high_resource_da_classifier.sh
    sh high_resource_es_classifier.sh
    sh high_resource_fi_classifer.sh
    sh high_resource_ru_classifer.sh
```

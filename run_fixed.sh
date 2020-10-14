#!/bin/bash
# remember to add --save when you want to save the experiment log

GPU=$1

DATASET=$3
MODELS=$2
ARGS="$4 $5 $6"
SAMPLES=(1.0)
TN=("20")
HIDG=("100:10")

SEED_ARRAY=(633 128 293 238 1372)
DATASEED_ARRAY=(-1 -1 -1 -1 -1)

if [ ${MODELS} == 'GRCN' ]; then
    if [ ${DATASET} == "Cora" ]; then
        PARA=("150")
    elif [ ${DATASET} == "CiteSeer" ]; then
        PARA=("300")
    else
        PARA=("1")
    fi
elif [[ ${MODELS} == *"GRCN"* ]]; then
    if [ ${DATASET} == "Cora" ]; then
        PARA=("100")
    elif [ ${DATASET} == "CiteSeer" ]; then
        PARA=("200")
    else
        PARA=("1")
    fi
fi


  for tn in "${TN[@]}"
  do
      for sample in "${SAMPLES[@]}"
      do
          for para in "${PARA[@]}"
          do
              for hidg in "${HIDG[@]}"
              do
                  for i in {0..4}
                  do
                      SEED=${SEED_ARRAY[i]}
                      DATASEED=${DATASEED_ARRAY[i]}
                      option="--dataset ${DATASET} --sample ${sample} --complete ${MODELS}
                          --compl_param ${para} --seed ${SEED} --dataseed ${DATASEED} ${ARGS}
                          --hid_graph ${hidg} --train_num ${tn}"
                      cmd="CUDA_VISIBLE_DEVICES=${GPU} python main_ours.py ${option}"
                      echo $cmd
                      eval $cmd
                  done
              done
          done
      done
  done

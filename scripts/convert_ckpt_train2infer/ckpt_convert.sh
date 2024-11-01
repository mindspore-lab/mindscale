#!/bin/bash

# help information
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -f,  --function              Convert function ('train_to_infer', 'quant_weight', 'distributed_weight_transfer' or 'pt_to_ms')"
    echo "  -p,  --precision             Set precision (fp16, w8a16, w8a8, w8a16c8, w8a8c8, fp16c8)"
    echo "  -w,  --world_size            Set the world size for distributed training (2, 4 or 8)"
    echo "  -y,  --yaml_path             Yaml path or model config path"
    echo "  -d,  --dataset_name          Dataset for convert w8a8 weights (boolq, squad1.1 and wikitext2)"
    echo "  -dp, --dataset_path          Dataset path (e.g. './boolq/dev.jsonl' )"
    echo "  -ts, --train_strategy_file   Training strategy saved path"
    echo "  -is, --infer_strategy_file   Inference strategy saved path"
    echo "  -sc, --src_ckpt_path         Source ckpt path"
    echo "  -dc, --dst_ckpt_path         Destination ckpt path"
    echo "  -pp, --pipeline_stage        Pipeline_stage set during training "
    echo "  -tm, --train_to_infer_mode    Train to infer mode ('1':ckpt_to_ckpt or '2':ckpt_to_safetensors_to_ckpt) "
    echo "  -h,  --help                  Print this help message"
}

# parser args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--function) function="$2"; shift ;;
        -p|--precision) precision="$2"; shift ;;
        -w|--world_size) world_size="$2"; shift ;;
        -y|--yaml_path) yaml_path="$2"; shift ;;
        -d|--dataset_name) dataset_name="$2"; shift ;;
        -dp|--dataset_path) dataset_path="$2"; shift ;;
        -ts|--train_strategy_file) train_strategy_file="$2"; shift ;;
        -is|--infer_strategy_file) infer_strategy_file="$2"; shift ;;
        -sc|--src_ckpt_path) src_ckpt_path="$2"; shift ;;
        -dc|--dst_ckpt_path) dst_ckpt_path="$2"; shift ;;
        -pp|--pipeline_stage) pipeline_stage="$2"; shift ;;
        -tm|--train_to_infer_mode) train_to_infer_mode="$2"; shift ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
    shift
done

# check input is not none
if [ -z "$precision" ] || [ -z "$world_size" ] ||  [ -z "$function" ]; then
    echo "Error: Missing precision, world_size and function required options."
    print_usage
    exit 1
fi

# print
echo "Settings:"
echo "  Precision: $precision"
echo "  World size: $world_size"
echo "  Function: $function"

export MS_COMPILER_CACHE_ENABLE=0
unset MS_DEV_RUNTIME_CONF

start_time=$(date +%H:%M:%S)

if [ -e "./log/" ]; then
    echo "./log/ is exit"
else
    mkdir ./log/
fi

n_to_m_rank_transformer(){ #Infer_strategy_path #Dst_ckpt_path #src_ckpt_path
    local src_ckpt_path=$1
    local Dst_ckpt_path=$2
    local Infer_strategy_path=$3
    local qkv_ffn=$4
    #1. 生成dir_count的strategy
    echo "----- Start to generate ${dir_count}p ${precision} strategy time: $(date +%H:%M:%S) -----"
    msrun --worker_num=$dir_count --local_worker_num=$dir_count \
    --log_dir=./log/msrun_log_save_strategy_${precision}_${dir_count}p --master_port=8126 \
    --join=True --bind_core=False \
    python save_strategy.py \
    --yaml_file=$yaml_path \
    --save_strategy_path=${Infer_strategy_path}_${dir_count}p \
    --world_size=$dir_count \
    --qkv_concat=${qkv_ffn} \
    --precision=${precision} \
    > ./log/log_save_strategy_${precision}_${dir_count}.log 2>&1
    echo "----- End generate ${dir_count}p ${precision} strategy time: $(date +%H:%M:%S) -----"
    #2. 生成world_size的strategy
    echo "----- Start to generate ${world_size}p ${precision} strategy time: $(date +%H:%M:%S) -----"
    msrun --worker_num=$world_size --local_worker_num=$world_size \
    --log_dir=./log/msrun_log_save_strategy_${precision}_${world_size}p --master_port=8126 \
    --join=True --bind_core=False \
    python save_strategy.py \
    --yaml_file=$yaml_path \
    --save_strategy_path=${Infer_strategy_path}_${world_size}p \
    --world_size=$world_size \
    --qkv_concat=${qkv_ffn} \
    --precision=${precision} \
    > ./log/log_save_strategy_${precision}_${world_size}.log 2>&1
    echo "----- End generate ${world_size}p ${precision} strategy time: $(date +%H:%M:%S) -----"
    #3. 转成worldsize的权重
    echo "----- Start to convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
    python transform_ckpt.py \
    --src_ckpt_strategy=${Infer_strategy_path}_${dir_count}p/strategy/ckpt_strategy_rank_0.ckpt \
    --dst_ckpt_strategy=${Infer_strategy_path}_${world_size}p/strategy/ckpt_strategy_rank_0.ckpt \
    --src_ckpt_dir=${src_ckpt_path} \
    --dst_ckpt_dir=${Dst_ckpt_path}_${world_size}p \
    --prefix="checkpoint_" \
    > ./log/log_transform_ckpt_${precision}_${world_size}.log 2>&1
    echo "----- End convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
    local result=$(awk "BEGIN {print $dir_count / $world_size==2}")
    if [ "$qkv_ffn" == "True" ] && [ "$result" -eq 1 ]; then
        echo "----- Start to adjust qkv and ffn time: $(date +%H:%M:%S)-----"
        python adjust_qkv.py \
            --src_ckpt_path=${Dst_ckpt_path}_${world_size}p \
            --dst_ckpt_path=${Dst_ckpt_path}_${world_size}p_adjust \
            --dir_count=${dir_count} \
            --world_size=${world_size} \
            --quant="$([ "$precision" != "fp16" ] && echo "True" || echo "False")" \
            > ./log/log_adjust_qkv_${dir_count}_to_${world_size}.log 2>&1
        echo "----- End convert qkv time: $(date +%H:%M:%S)-----"
    fi
}

1_to_m_rank_transformer(){ #Infer_strategy_path #Dst_ckpt_path #src_ckpt_path
    local src_ckpt_path=$1
    local Dst_ckpt_path=$2
    local Infer_strategy_path=$3
    local qkv_ffn=$4
    #2. 生成world_size的strategy
    echo "----- Start to generate ${world_size}p ${precision} strategy time: $(date +%H:%M:%S) -----"
    msrun --worker_num=$world_size --local_worker_num=$world_size \
    --log_dir=./log/msrun_log_save_strategy_${precision}_${world_size}p --master_port=8126 \
    --join=True --bind_core=False \
    python save_strategy.py \
    --yaml_file=$yaml_path \
    --save_strategy_path=${Infer_strategy_path}_${world_size}p \
    --world_size=$world_size \
    --qkv_concat=${qkv_ffn} \
    --precision=${precision} \
    > ./log/log_save_strategy_${precision}_${world_size}.log 2>&1
    echo "----- End generate ${world_size}p ${precision} strategy time: $(date +%H:%M:%S) -----"
    #3. 转成worldsize的权重
    echo "----- Start to convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
#    python transform_ckpt.py \
#    --dst_ckpt_strategy=${Infer_strategy_path}_${world_size}p/strategy/ckpt_strategy_rank_0.ckpt \
#    --src_ckpt_dir=${src_ckpt_path}  \
#    --dst_ckpt_dir=${Dst_ckpt_path}_${world_size}p \
#    --prefix="checkpoint_" \
    python transform_ckpt_safetensors.py \
    --dst_ckpt_strategy=${Infer_strategy_path}_${world_size}p/strategy  \
    --src_ckpt_dir=${src_ckpt_path}  \
    --dst_ckpt_dir=${Dst_ckpt_path}_${world_size}p \
    > ./log/log_transform_ckpt_${precision}_${world_size}.log 2>&1
    echo "----- End convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
}

distributed_weight_transfer(){  #Infer_strategy_path #Dst_ckpt_path #src_ckpt_path
    local src_ckpt_path=$1
    local Dst_ckpt_path=$2
    local Infer_strategy_path=$3
    local quant="$([ "$precision" != "fp16" ] && echo 'True' || echo 'False')"
    local result=$(awk "BEGIN {print $dir_count / $world_size==2}")
    echo "dir_count/world_size==2: $result"
    check_qkv_output_1=$(python check_weight_name.py --src_ckpt_dir=$src_ckpt_path)
    echo "Python 脚本的输出是: $check_qkv_output_1"
    if echo "$check_qkv_output_1" | grep -q 'yes-qkv' ; then
        echo "qkv_ffn=True"
        qkv_ffn="True"
    else
        echo "qkv_ffn=False"
        qkv_ffn="False"
    fi
    if [ "$quant" == "True" ] && [ "$result" -ne 1 ]; then
       echo "Not support this input combination."
       exit 1
    fi
    if [ "$quant" == "True" ] && [ "$qkv_ffn" == "False" ] && [ "$result" -eq 1 ]; then
        echo "Not support this input combination."
        exit 1
    fi
    if [ "$dir_count" == 1 ]; then
        if [ "$qkv_ffn" == 'True' ]; then
            echo "----- Start to revert qkv and ffn time: $(date +%H:%M:%S)-----"
            python revert_qkv_ffn.py \
            --world_size=$dir_count \
            --src_ckpt_path=${src_ckpt_path}  \
            --dst_ckpt_path=${src_ckpt_path}_no_qkv \
            > ./log/log_revert_qkv_ffn.log 2>&1
            echo "----- End revert qkv time: $(date +%H:%M:%S)-----"
            qkv_ffn="False"
            1_to_m_rank_transformer ${src_ckpt_path}_no_qkv ${Dst_ckpt_path} ${Infer_strategy_path} ${qkv_ffn}
        else
            1_to_m_rank_transformer ${src_ckpt_path} ${Dst_ckpt_path} ${Infer_strategy_path} ${qkv_ffn}
        fi
    else
        if [ "$qkv_ffn" == 'True' ] && [ "$result" -ne 1 ]; then
            echo "----- Start to revert qkv and ffn time: $(date +%H:%M:%S)-----"
            python revert_qkv_ffn.py \
            --world_size=$dir_count \
            --src_ckpt_path=${src_ckpt_path}  \
            --dst_ckpt_path=${src_ckpt_path}_no_qkv \
            > ./log/log_revert_qkv_ffn.log 2>&1
            echo "----- End revert qkv time: $(date +%H:%M:%S)-----"
            qkv_ffn="False"
            n_to_m_rank_transformer ${src_ckpt_path}_no_qkv ${Dst_ckpt_path} ${Infer_strategy_path} ${qkv_ffn}
        else
            n_to_m_rank_transformer ${src_ckpt_path} ${Dst_ckpt_path} ${Infer_strategy_path} ${qkv_ffn}
        fi
    fi
    if [ "$qkv_ffn" == "False" ] && [ "$precision" == 'fp16' ]; then
        echo "----- Start to convert qkv and ffn time: $(date +%H:%M:%S)-----"
        python convert_qkv_ffn.py \
        --world_size=$world_size \
        --src_ckpt_path=${Dst_ckpt_path}_${world_size}p  \
        --dst_ckpt_path=${Dst_ckpt_path}_${world_size}p_qkv \
        > ./log/log_convert_qkv_ffn.log 2>&1
        echo "----- End convert qkv time: $(date +%H:%M:%S)-----"
    fi
}


# train_to_infer_fp16
if [ "$function" == "train_to_infer" ] && [ "$precision" == "fp16" ]; then
    echo "Converting checkpoint from training to inference, selected precision: $precision."
    # fp16 train_2_infer
    if [ -z "$infer_strategy_file" ]; then
       Infer_strategy_path=./infer_strategy/${precision}_${world_size}p
    else
       Infer_strategy_path=${infer_strategy_file}/${precision}_${world_size}p
    fi
    #1. save strategy
    echo "-----1. Start to save strategy time: $(date +%H:%M:%S) -----"
    msrun --worker_num=$world_size --local_worker_num=$world_size \
    --log_dir=./log/msrun_log_save_strategy_${precision}_${world_size}p \
    --master_port=8126 --join=True --bind_core=False \
    python save_strategy.py \
    --yaml_file=$yaml_path \
    --save_strategy_path=${Infer_strategy_path}_no_qkv \
    --world_size=$world_size \
    --qkv_concat=False \
    --precision=${precision} \
    > ./log/log_save_strategy_${precision}_${world_size}p.log 2>&1
    echo "-----1. End save strategy  time: $(date +%H:%M:%S) -----"
    if [ -z "$src_ckpt_path" ] || [ -z "$train_strategy_file" ] ; then
        echo "Please set training checkpoint saved path and training strategy saved path."
        exit 1
    fi
    if [ -z "$dst_ckpt_path" ]; then
       Infer_ckpt_path=./infer_ckpt/${precision}_${world_size}p
    else
       Infer_ckpt_path=${dst_ckpt_path}/${precision}_${world_size}p
    fi
    #2. 训练转推理
    echo "-----2. Start to convert train to infer weights time: $(date +%H:%M:%S) -----"
    if [ "$train_to_infer_mode" == "2" ]; then
        python train2infer_safetensors.py \
        --train_ckpt_path=$src_ckpt_path \
        --del_optim_path=$Infer_ckpt_path"/del_optim" \
        --train_strategy_file=$train_strategy_file \
        --infer_strategy_file=${Infer_strategy_path}_no_qkv/strategy \
        --train_2_infer_path=$Infer_ckpt_path"/train_2_infer_ckpt" \
        --infer_ckpt_path=$Infer_ckpt_path > ./log/log_train_2_infer_${world_size}.log 2>&1
    else
        python train2infer.py \
        --train_ckpt_path=$src_ckpt_path \
        --del_optim_path=$Infer_ckpt_path"/del_optim" \
        --train_strategy_file=$train_strategy_file \
        --infer_strategy_file=${Infer_strategy_path}_no_qkv/strategy \
        --train_2_infer_path=$Infer_ckpt_path"/train_2_infer_ckpt" \
        --infer_ckpt_path=$Infer_ckpt_path \
        --world_size=$world_size \
        --pipeline_stage=$pipeline_stage > ./log/log_train_2_infer_${world_size}.log 2>&1
    fi
    echo "-----2. End convert train to infer weights time: $(date +%H:%M:%S) -----"
    #3. 增加前端融合算子
    echo "-----3. Start to convert qkv and ffn time: $(date +%H:%M:%S)-----"
    python convert_qkv_ffn.py \
    --world_size=$world_size \
    --src_ckpt_path=$Infer_ckpt_path \
    --dst_ckpt_path=$Infer_ckpt_path"_qkv" \
    > ./log/log_convert_qkv_ffn.log 2>&1
    echo "-----3. End convert qkv time: $(date +%H:%M:%S)-----"
    rm -rf $Infer_ckpt_path"/del_optim"
    rm -rf $Infer_ckpt_path"/train_2_infer_ckpt"
elif [ "$function" == "quant_weight" ]; then
    echo "Converting checkpoint from fp16 inference to ${precision} inference"
    # 检查输入值
    if [ -z "$src_ckpt_path" ] || [ -z "$yaml_path" ] ; then
        echo "Please set fp16 checkpoint saved path and ${precision} yaml file"
        exit 1
    fi
    if [ -z "$dst_ckpt_path" ]; then
       Dst_ckpt_path=./infer_ckpt/${precision}
    else
       Dst_ckpt_path=${dst_ckpt_path}/${precision}
    fi
    if [ -z "$infer_strategy_file" ]; then
       Infer_strategy_path=./infer_strategy/${precision}
    else
       Infer_strategy_path=${infer_strategy_file}/${precision}
    fi
    # 检查是否是相同卡数
    dir_count=$(find  "$src_ckpt_path"  -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Number of directories in '$src_ckpt_path' is: $dir_count"
    if [ "$dir_count" != "$world_size" ] ; then
        rank_num=$dir_count
    else
        rank_num=$world_size
    fi
    if [ "$precision" == "w8a16" ]; then
        #1. 转换成rank_num的w8a16权重
        echo "----- Start to convert ${rank_num}p ${precision} weights time: $(date +%H:%M:%S) -----"
        msrun --worker_num=$rank_num --local_worker_num=$rank_num --master_port=8126 \
        --log_dir=./log/msrun_log_${precision}_${rank_num}_ckpt --join=True --cluster_time_out=300 \
        python quant_ckpt.py \
        -c $yaml_path \
        -q ptq \
        -t $dataset_name \
        -s $dataset_path \
        -w int8 \
        -a None \
        -k None \
        -o None \
        -b lm_head \
        -lc $src_ckpt_path  \
        -od ${Dst_ckpt_path}_${rank_num}p \
        -ws $rank_num \
        > ./log/log_fp16_to_${precision}_${rank_num}.log 2>&1
        if find "${Dst_ckpt_path}_${rank_num}p/rank_0/" -type f -name "*.ckpt" | read; then
            echo "----- End convert ${rank_num}p ${precision} weights time: $(date +%H:%M:%S) -----"
        else
            echo "QUANT ERROR"
            exit 1
        fi
    elif [ "$precision" == "w8a8" ]; then
        #1. 转换成rank_num的w8a8权重
        echo "----- Start to convert ${rank_num}p ${precision} weights time: $(date +%H:%M:%S) -----"
        msrun --worker_num=$rank_num --local_worker_num=$rank_num --master_port=8126 \
        --log_dir=./log/msrun_log_${precision}_${rank_num}_ckpt --join=True --cluster_time_out=300 \
        python quant_ckpt.py \
        -c $yaml_path \
        -q ptq \
        -t $dataset_name \
        -s $dataset_path \
        -w int8 \
        -a int8 \
        -k None \
        -o smooth \
        -b lm_head \
        -lc $src_ckpt_path  \
        -od ${Dst_ckpt_path}_${rank_num}p \
        -ws $rank_num \
        > ./log/log_fp16_to_${precision}_${rank_num}.log 2>&1
        if find "${Dst_ckpt_path}_${rank_num}p/rank_0/" -type f -name "*.ckpt" | read; then
            echo "----- End convert ${rank_num}p ${precision} weights time: $(date +%H:%M:%S) -----"
        else
            echo "QUANT ERROR"
            exit 1
        fi
    elif [ "$precision" == "w8a16c8" ]; then
        #1. 转换成rank_num的w8a16c8权重
        echo "----- Start to convert ${rank_num}p ${precision} weights time: $(date +%H:%M:%S) -----"
        msrun --worker_num=$rank_num --local_worker_num=$rank_num --master_port=8126 \
        --log_dir=./log/msrun_log_${precision}_${rank_num}_ckpt --join=True --cluster_time_out=300 \
        python quant_ckpt.py \
        -c $yaml_path \
        -q ptq \
        -t $dataset_name \
        -s $dataset_path \
        -w int8 \
        -a None \
        -k int8 \
        -o None \
        -b lm_head \
        -lc $src_ckpt_path  \
        -od ${Dst_ckpt_path}_${rank_num}p \
        -ws $rank_num \
        > ./log/log_fp16_to_${precision}_${rank_num}.log 2>&1
        if find "${Dst_ckpt_path}_${rank_num}p/rank_0/" -type f -name "*.ckpt" | read; then
            echo "----- End convert ${rank_num}p ${precision} weights time: $(date +%H:%M:%S) -----"
        else
            echo "QUANT ERROR"
            exit 1
        fi
    elif [ "$precision" == "w8a8c8" ]; then
        #1. 转换成rank_num的w8a8c8权重
        echo "----- Start to convert ${rank_num}p ${precision} weights time: $(date +%H:%M:%S) -----"
        msrun --worker_num=$rank_num --local_worker_num=$rank_num --master_port=8126 \
        --log_dir=./log/msrun_log_${precision}_${rank_num}_ckpt --join=True --cluster_time_out=300 \
        python quant_ckpt.py \
        -c $yaml_path \
        -q ptq \
        -t $dataset_name \
        -s $dataset_path \
        -w int8 \
        -a int8 \
        -k int8 \
        -o smooth \
        -b lm_head \
        -lc $src_ckpt_path  \
        -od ${Dst_ckpt_path}_${rank_num}p \
        -ws $rank_num \
        > ./log/log_fp16_to_${precision}_${rank_num}.log 2>&1
        if find "${Dst_ckpt_path}_${rank_num}p/rank_0/" -type f -name "*.ckpt" | read; then
            echo "----- End convert ${rank_num}p ${precision} weights time: $(date +%H:%M:%S) -----"
        else
            echo "QUANT ERROR"
            exit 1
        fi
    elif [ "$precision" == "fp16c8" ]; then
        #1. 转换成rank_num的fp16c8权重
        echo "----- Start to convert ${rank_num}p ${precision} weights time: $(date +%H:%M:%S) -----"
        msrun --worker_num=$rank_num --local_worker_num=$rank_num --master_port=8126 \
        --log_dir=./log/msrun_log_${precision}_${rank_num}_ckpt --join=True --cluster_time_out=300 \
        python quant_ckpt.py \
        -c $yaml_path \
        -q ptq \
        -t $dataset_name \
        -s $dataset_path \
        -w None \
        -a None \
        -k int8 \
        -o None \
        -b None \
        -lc $src_ckpt_path  \
        -od ${Dst_ckpt_path}_${rank_num}p \
        -ws $rank_num \
        > ./log/log_fp16_to_${precision}_${rank_num}.log 2>&1
        if find "${Dst_ckpt_path}_${rank_num}p/rank_0/" -type f -name "*.ckpt" | read; then
            echo "----- End convert ${rank_num}p ${precision} weights time: $(date +%H:%M:%S) -----"
        else
            echo "QUANT ERROR"
            exit 1
        fi
    else
        echo "Wrong precision input"
        exit 1
    fi
    if [ "$dir_count" == "$world_size" ] ; then
        echo "Quantification weights is finish!"
        end_time=$(date +%H:%M:%S)
        echo "Total Start Time: $start_time, Total End Time: $end_time"
        exit 0
    else
        echo "Start to transfer weight from ${dir_count} to ${world_size}"
        distributed_weight_transfer ${Dst_ckpt_path}_${dir_count}p ${Dst_ckpt_path} ${Infer_strategy_path}
    fi
elif [ "$function" == "distributed_weight_transfer" ]; then
    echo "Distributed weight transfer for ${precision} inference"
    # 检查输入值
    if [ -z "$src_ckpt_path" ] || [ -z "$yaml_path" ] ; then
        echo "Please set checkpoint saved path and ${precision} yaml file"
        exit 1
    fi
    if [ -z "$dst_ckpt_path" ]; then
       Dst_ckpt_path=./infer_ckpt/${precision}
    else
       Dst_ckpt_path=${dst_ckpt_path}/${precision}
    fi
    if [ -z "$infer_strategy_file" ]; then
       Infer_strategy_path=./infer_strategy/${precision}
    else
       Infer_strategy_path=${infer_strategy_file}/${precision}
    fi
    # 检查是否是相同卡数
    dir_count=$(find  "$src_ckpt_path"  -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Number of directories in '$src_ckpt_path' is: $dir_count"
    if [ "$dir_count" == "$world_size" ] ; then
        echo "The input ckpt ranks equal to world size, no need to transfer"
    else
        distributed_weight_transfer ${src_ckpt_path} ${Dst_ckpt_path} ${Infer_strategy_path}
    fi
elif [ "$function" == "pt_to_ms" ]; then
    echo "Converting pytorch bin to mindspore ckpt"
    if [ "$precision" != "fp16" ]; then
      echo "Only support precision is fp16 in pytorch convert to mindspore "
      precision="fp16"
    fi
    if [ -z "$dst_ckpt_path" ]; then
       Dst_ckpt_path=./pt_to_ms/${precision}
    else
       Dst_ckpt_path=${dst_ckpt_path}/${precision}
    fi
    mkdir -p ${Dst_ckpt_path}_complete_${world_size}/rank_0/
    #1. bin to ckpt
    dir_count=1
    echo "----- Start to convert bin to ckpt time: $(date +%H:%M:%S)-----"
    python convert_pt2ms.py \
    --torch_ckpt_dir $src_ckpt_path \
    --mindspore_ckpt_file ${Dst_ckpt_path}_complete_${world_size}/rank_0/llama57b.ckpt \
    > ./log/log_pt_to_ms.log 2>&1
    echo "----- End convert bin to ckpt time: $(date +%H:%M:%S)-----"
    distributed_weight_transfer ${Dst_ckpt_path}_complete_${world_size} ${Dst_ckpt_path}_ckpt_${world_size} ${Dst_ckpt_path}_strategy
else
    echo "Your input mode of training to inference conversion is not supported."
fi

echo "Convert finish!"
end_time=$(date +%H:%M:%S)
echo "Total Start Time: $start_time, Total End Time: $end_time"

BASE_DIR=/mnt/disks/scratch1/concurrentqa/ # path to the repo directory
OUTPUT_PATH=${BASE_DIR}/RUNS/HOTPOT/privateprop_0.5/ # path to where the public and private question and corpora splits are located
ORIGINAL_DATA=/mnt/disks/scratch1/hotpot/hotpot_extras/hotpot_dev_distractor_v1.json
HOTPOT_MODELS_PATH=/mnt/disks/scratch1/hotpot_models/

# Settings
RUN_ID=0
NUM_RETRIEVED=100
RESTRICTED=0 # apply document privacy if 1; restrict private hop 1 to public hop 2 retrieval path
QUERY_PRIVACY=0 # apply query privacy if 1; only retrieve from local corpus
MIN_BY_DOMAIN=0 # assert a requirement to retrieve at least this number of passages per domain per hop

# RETRIEVAL MODES
# 4combo_separaterank: private and public documents are in 2 different corpora, retrieve separately and take the top k passage-chains ***from each*** of public/private, private/public, private/private, public/public chains for the reader
# 4combo_overallrank: private and public documents are in 2 different corpora, retrieve separately and take the top k passage-chains ***overall*** public/private, private/public, private/private, public/public chains for the reader
# fullindex: treat all private and public documents in one corpus and retrieve
# 2combo_oracledomain: retrieve with the knowledge of the gold supporting paths
RETRIEVAL_MODE=fullindex

# Which steps to run?
ENCODE=0
RETRIEVE_MULTI=0
RETRIEVE_FULL=1
READ=1

if [[ $ENCODE == 1 ]]; then
    for DOMAIN in 0 1;
	do
#     # MULTI-INDEX PASSAGE ENCODING
	    CUDA_VISIBLE_DEVICES=0,1,2,3 python encode_corpus.py \
	        --do_predict \
		--predict_batch_size 1000 \
		--model_name roberta-base \
		--predict_file ${OUTPUT_PATH}/domain${DOMAIN}psgs.json \
		--init_checkpoint ${HOTPOT_MODELS_PATH}/doc_encoder.pt \
		--embed_save_path ${OUTPUT_PATH}/domain_${DOMAIN}/ \
		--fp16 \
		--max_c_len 300 \
		--num_workers 20
	done
fi

# COMBINE ALL QUESTIONS AND RETRIEVE OVER EACH INDEX (i.e. distributed indices)
if [[ $RETRIEVE_MULTI == 1 ]]; then
    python eval_mhop_retrieval.py \
     	${OUTPUT_PATH}/hotpot_qas_val_all.json \
    	${OUTPUT_PATH}/domain_0/idx.npy \
     	${OUTPUT_PATH}/domain_0/id2doc.json \
     	${HOTPOT_MODELS_PATH}/q_encoder.pt \
     	--indexpath_alt ${OUTPUT_PATH}/domain_1/idx.npy \
     	--corpus_dict_alt ${OUTPUT_PATH}/domain_1/id2doc.json \
     	--batch-size 10 \
     	--beam-size ${NUM_RETRIEVED} \
     	--topk ${NUM_RETRIEVED} \
     	--shared-encoder \
     	--model-name roberta-base \
     	--retrieval_mode ${RETRIEVAL_MODE} \
     	--restricted ${RESTRICTED} \
        --query_privacy ${QUERY_PRIVACY} \
     	--gpu \
        --gpu_num 2 \
     	--save-path ${OUTPUT_PATH}/retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_RESTRICTED${RESTRICTED}.json \
     	--metrics-path ${OUTPUT_PATH}/retrieval_metrics_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_RESTRICTED${RESTRICTED}.json
fi



# COMBINE ALL QUESTIONS AND ALL DOCUMENTS, AND RETRIEVE (i.e. NO PRIVACY)
if [[ $RETRIEVE_FULL == 1 ]]; then
    python eval_mhop_retrieval.py \
     	${OUTPUT_PATH}/hotpot_qas_val_all.json \
    	${BASE_DIR}/DATA/wikionly/idx.npy \
     	${BASE_DIR}/DATA/wikionly/id2doc.json \
     	${HOTPOT_MODELS_PATH}/q_encoder.pt \
     	--batch-size 10 \
     	--beam-size ${NUM_RETRIEVED} \
     	--topk ${NUM_RETRIEVED} \
     	--shared-encoder \
     	--model-name roberta-base \
     	--retrieval_mode ${RETRIEVAL_MODE} \
     	--restricted ${RESTRICTED} \
        --query_privacy ${QUERY_PRIVACY} \
     	--gpu \
        --gpu_num 2 \
     	--save-path ${OUTPUT_PATH}/retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_RESTRICTED${RESTRICTED}.json \
     	--metrics-path ${OUTPUT_PATH}/retrieval_metrics_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_RESTRICTED${RESTRICTED}.json
fi


if [[ $READ == 1 ]]; then
    RETRIEVED_DATA=${OUTPUT_PATH}/retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_RESTRICTED${RESTRICTED}.json
    SAVED_PATH=${OUTPUT_PATH}/retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_RESTRICTED${RESTRICTED}_w_sp.json
    python mdr/retrieval/utils/mhop_utils.py ${ORIGINAL_DATA} ${RETRIEVED_DATA} ${SAVED_PATH}

    # RUN QA READING on EACH INDEX (SINGLE-RETRIEVAL)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_qa.py \
        --do_predict \
        --predict_batch_size 200 \
        --model_name google/electra-large-discriminator \
        --fp16 \
        --predict_file ${OUTPUT_PATH}/retrieval_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_RESTRICTED${RESTRICTED}_w_sp.json \
        --max_seq_len 512 \
        --max_q_len 64 \
        --init_checkpoint ${HOTPOT_MODELS_PATH}/qa_electra.pt \
        --sp-pred \
        --max_ans_len 30 \
        --save-prediction ${OUTPUT_PATH}/hotpot_val_reader_results_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_RESTRICTED${RESTRICTED}.json \
        --save_raw_results ${OUTPUT_PATH}/hotpot_val_reader_rawanswers_num${NUM_RETRIEVED}_mode${RETRIEVAL_MODE}_RESTRICTED${RESTRICTED}.json
fi

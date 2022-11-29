CUDA_VISIBEL_DEVICES='0' python ./predict_cail22_xxcq_roformer_negative_learning.py \
    --input_file '/data/albert.xht/unified_information_extraction/CAIL2022/xxcq/datasets/step2_test.json' \
    --config_file 'config/config_cail22_xxcq_roformer.ini' \
    --output_file 'final_test/roformer_negative_learing.json'
    
# CUDA_VISIBEL_DEVICES='0' python ./predict_seq2struct_unilm_cail22_xxcq.py \
#     --input_file '/data/albert.xht/unified_information_extraction/CAIL2022/xxcq/datasets/step2_test.json' \
#     --config_file 'config/config_unilm_roformer_prediction_cail_xxcq_v4.ini' \
#     --output_file 'final_test/unilm_roformer_large_v5.json'

# CUDA_VISIBEL_DEVICES='0' python ./predict_baffine.py \
#     --input_file '/data/albert.xht/unified_information_extraction/CAIL2022/xxcq/datasets/step2_test.json' \
#     --config_file 'model_save/ie_baffine_v2.ini' \
#     --output_file 'final_test/ie_baffine_v2.json'
    
# CUDA_VISIBEL_DEVICES='0' python ./predict_baffine.py \
#     --input_file '/data/albert.xht/unified_information_extraction/CAIL2022/xxcq/datasets/step2_test.json' \
#     --config_file 'model_save/ie_baffine_v1.ini' \
#     --output_file 'final_test/ie_baffine_v1.json'
    
# python ./vote.py \
#     --input_folder 'final_test' \
#     --majority_vote 2 \
#     --output_file './'
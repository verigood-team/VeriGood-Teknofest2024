from mini_pyabsa import AspectSentimentTripletExtraction as ASTE
from mini_pyabsa.utils.data_utils.dataset_item import DatasetItem

if __name__ == "__main__":
    config = ASTE.ASTEConfigManager.get_aste_config_multilingual()
    config.model = ASTE.ASTEModelList.EMCGCN
    config.max_seq_len = 256
    config.spacy_model = "tr_core_news_lg"
    config.log_step = -1
    config.pretrained_bert = "microsoft/mdeberta-v3-base"
    config.num_epoch = 25
    config.batch_size = 1
    config.learning_rate = 1e-5
    # config.load_cache_path = r"emcgcn.1.custom.dataset.a0350b0a1d4c091be8b2489a722478721c70b67c47531e1830b760069535681e.cache"

    dataset = DatasetItem(r"1.custom")

    trainer = ASTE.ASTETrainer(
        config=config,
        dataset=dataset,
        from_checkpoint=r"checkpoints\dataset_45.31",
        checkpoint_save_mode=1,
        auto_device=True,
    )
    
    triplet_extractor = trainer.load_trained_model()

    examples = ["Example1", "Example2"]
    
    for i in range(1000):
        triplet_extractor.predict(input())



# MANUEL TEST

# triplet_extractor = ASTE.AspectSentimentTripletExtractor(r"checkpoints\all_part4_44.77")

# examples = ["Vodafone hem ucuz hem de her yerde çekiyor herkese tavsiye ederim",]
    
# for i in range(1000):
#     entity_list, results = triplet_extractor.predict(input())
#     # print("entity_list",entity_list)
#     # print("results",results)
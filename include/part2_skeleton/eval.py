from ranking import ranking


def mapk(caption_embeddings, image_embeddings, caption_labels, image_labels, mode='desc', k=10):
    if mode == 'img':
        ranking_images = ranking.rank_embedding(
            caption_embed=caption_embeddings,
            caption_id=caption_labels,
            image_embed=image_embeddings,
            image_id=image_labels,
            retrieve="images",
            distance_metric="Hamming",
            k=10,
            add_correct_id=True
        )
        average_precision_images = ranking.average_precision(ranking_images, gtp=1)
        return round(average_precision_images.mean()[0] * 100, 10)
    elif mode == 'desc':
        ranking_captions = ranking.rank_embedding(
            caption_embed=caption_embeddings,
            caption_id=caption_labels,
            image_embed=image_embeddings,
            image_id=image_labels,
            retrieve="captions",
            distance_metric="Hamming",
            k=10,
            add_correct_id=True
        )
        average_precision_captions = ranking.average_precision(ranking_captions, gtp=5)
        return round(average_precision_captions.mean()[0] * 100, 10)
    else:
        print('invalid mapk mode: {} , should be in desc or img'.format(mode))
        return 0

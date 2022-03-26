camera_positions = ['Camera Left (Looking Right)', 'Camera Right (Looking Left)',
                    'Camera up (Looking Down)', 'Camera down (Looking up)',
                    'Forward']

# cat_6 = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
cat_6 = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

cat_26 = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence',
          'Disapproval', 'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment',
          'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace',
          'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
cat_map = {
    'Anger': ['Anger']
    , 'Disgust': ['Aversion']
    , 'Fear': ['Fear']
    , 'Happiness': ['Happiness', 'Pleasure']
    , 'Sadness': ['Sadness']
    , 'Surprise': ['Surprise']
}

def discrete_categories(ind2cat=True, categories_number=6):
    v_cat_6 = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    v_cat_26 = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence',
              'Disapproval', 'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment',
              'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace',
              'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    cat = v_cat_6 if categories_number == 6 else v_cat_26
    d_cat2ind = {}
    d_ind2cat = {}
    for idx, emotion in enumerate(cat):
        d_cat2ind[emotion] = idx
        d_ind2cat[idx] = emotion
    return d_ind2cat if ind2cat else d_cat2ind

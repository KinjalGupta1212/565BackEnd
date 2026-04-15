import numpy as np
import pandas as pd
import pymc as pm
import preprocess


# =========================================================
# 1. WIDE → LONG FORMAT
# =========================================================

def wide_to_long(df, item_cols, annotator_col, comment_col):
    rows = []

    for _, row in df.iterrows():
        for item_id, col in enumerate(item_cols):
            rows.append({
                "comment_id": row[comment_col],
                "annotator_id": row[annotator_col],
                "item_id": item_id,
                "response": row[col]
            })

    return pd.DataFrame(rows)


# =========================================================
# 2. REINDEX IDS + KEEP MAPS
# =========================================================

def reindex_ids(df):
    df = df.copy()

    df["comment_id"], comment_map = pd.factorize(df["comment_id"])
    df["annotator_id"], annotator_map = pd.factorize(df["annotator_id"])
    df["item_id"], item_map = pd.factorize(df["item_id"])

    return df, comment_map, annotator_map, item_map


# =========================================================
# 3. FAST FACETED RASCH (ADVI INFERENCE)
# =========================================================

def fit_rasch(data):
    comment_id = data["comment_id"].values
    item_id = data["item_id"].values
    annotator_id = data["annotator_id"].values
    response = data["response"].values

    n_comments = len(np.unique(comment_id))
    n_items = len(np.unique(item_id))
    n_annotators = len(np.unique(annotator_id))

    with pm.Model() as model:

        # latent variables
        theta = pm.Normal("theta", 0, 1, shape=n_comments)
        beta = pm.Normal("beta", 0, 1, shape=n_items)
        alpha = pm.Normal("alpha", 0, 1, shape=n_annotators)

        # linear predictor
        eta = (
            theta[comment_id]
            - beta[item_id]
            - alpha[annotator_id]
        ) * 0.5

        # cutpoints
        cutpoints = pm.Normal(
            "cutpoints",
            0,
            1,
            shape=4,
            transform=pm.distributions.transforms.ordered,
            initval=np.array([-1.5, -0.5, 0.5, 1.5])
        )

        pm.OrderedLogistic(
            "obs",
            eta=eta,
            cutpoints=cutpoints,
            observed=response
        )

        # =====================================================
        # ⚡ FAST INFERENCE (KEY CHANGE)
        # =====================================================
        approx = pm.fit(
            n=50_000,
            method="advi",
            obj_n_mc=1
        )

        trace = approx.sample(1000)

    return model, trace


# =========================================================
# 4. EXTRACT SCORES
# =========================================================

def get_scores(trace):
    return trace.posterior["theta"].mean(dim=("chain", "draw")).values


# =========================================================
# 5. LOOKUP FUNCTION
# =========================================================

def get_comment_score(trace, comment_map, original_id):
    theta = get_scores(trace)

    idx = np.where(comment_map == original_id)[0][0]
    return theta[idx]


# =========================================================
# 6. MAIN
# =========================================================

if __name__ == "__main__":

    df = preprocess.load_data()

    item_cols = [
        "sentiment", "respect", "insult", "humiliate", "status",
        "dehumanize", "violence", "genocide", "attack_defend", "hatespeech"
    ]

    # convert wide → long
    long_df = wide_to_long(
        df,
        item_cols=item_cols,
        annotator_col="annotator_id",
        comment_col="comment_id"
    )

    # reindex + maps
    long_df, comment_map, annotator_map, item_map = reindex_ids(long_df)

    print("LONG DF TYPE:", type(long_df))
    print(long_df.head())

    # fit model (FAST)
    model, trace = fit_rasch(long_df)

    # extract theta
    theta = get_scores(trace)

    # =====================================================
    # 🔥 GET COMMENT ID = 4
    # =====================================================

    score_4 = get_comment_score(trace, comment_map, 4)

    print("\n==============================")
    print("HATE SPEECH SCORE (COMMENT 4)")
    print("==============================")
    print(score_4)

    # optional comparison
    if "hate_speech_score" in df.columns:
        true_score = df[df["comment_id"] == 4]["hate_speech_score"].values[0]

        print("\nDATASET SCORE:", true_score)
        print("DIFFERENCE:", score_4 - true_score)
import numpy as np
import pandas as pd
import pymc as pm
import preprocess


# =========================================================
# 1. WIDE → LONG
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
# 2. REINDEX + KEEP MAPS (SAFE)
# =========================================================

def reindex_ids(df):
    df = df.copy()

    df["comment_id"], comment_map = pd.factorize(df["comment_id"])
    df["annotator_id"], annotator_map = pd.factorize(df["annotator_id"])
    df["item_id"], item_map = pd.factorize(df["item_id"])

    return df, comment_map, annotator_map, item_map


# =========================================================
# 3. FACETED RASCH (STABLE)
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

        theta_raw = pm.Normal("theta_raw", 0, 1, shape=n_comments)
        beta_raw = pm.Normal("beta_raw", 0, 1, shape=n_items)
        alpha_raw = pm.Normal("alpha_raw", 0, 1, shape=n_annotators)

        # identification
        theta = pm.Deterministic("theta", theta_raw - pm.math.mean(theta_raw))
        beta = pm.Deterministic("beta", beta_raw - pm.math.mean(beta_raw))
        alpha = pm.Deterministic("alpha", alpha_raw - pm.math.mean(alpha_raw))

        eta = (theta[comment_id] - beta[item_id] - alpha[annotator_id]) / 2.0

        cutpoints = pm.Normal(
            "cutpoints",
            mu=np.array([-1.5, -0.5, 0.5, 1.5]),
            sigma=0.5,
            shape=4,
            transform=pm.distributions.transforms.ordered
        )

        pm.OrderedLogistic(
            "obs",
            eta=eta,
            cutpoints=cutpoints,
            observed=response
        )

        approx = pm.fit(
            n=30_000,
            method="advi",
            obj_n_mc=1
        )

        trace = approx.sample(1000)

    return model, trace


# =========================================================
# 4. EXTRACT THETA
# =========================================================

def get_theta(trace):
    return trace.posterior["theta"].mean(("chain", "draw")).values


# =========================================================
# 5. HF ALIGNMENT (NO LEAKAGE FIX)
# =========================================================

def align_to_hf_scale(theta, hf_df, comment_map):

    # compute HF only for comments actually in model
    valid_ids = [cid for cid in comment_map]

    hf = hf_df.groupby("comment_id")["hate_speech_score"].mean()
    hf = hf[hf.index.isin(valid_ids)]

    if len(hf) == 0:
        return theta

    theta = (theta - theta.mean()) / (theta.std() + 1e-8)
    theta = theta * hf.std() + hf.mean()

    return theta


# =========================================================
# 6. SAFE LOOKUP
# =========================================================

def get_comment_score(theta, comment_map, original_id):

    matches = np.where(comment_map == original_id)[0]

    if len(matches) == 0:
        raise ValueError(
            f"comment_id {original_id} not in training sample. "
            "Increase sample size or use full dataset."
        )

    return theta[matches[0]]


# =========================================================
# 7. MAIN
# =========================================================
if __name__ == "__main__":

    df = preprocess.load_data()

    # sample dataset
    df = df.sample(5000, random_state=42)

    item_cols = [
        "sentiment", "respect", "insult", "humiliate", "status",
        "dehumanize", "violence", "genocide", "attack_defend", "hatespeech"
    ]

    long_df = wide_to_long(
        df,
        item_cols=item_cols,
        annotator_col="annotator_id",
        comment_col="comment_id"
    )

    long_df, comment_map, annotator_map, item_map = reindex_ids(long_df)

    print("LONG DF SIZE:", len(long_df))

    model, trace = fit_rasch(long_df)

    theta = get_theta(trace)
    theta_aligned = align_to_hf_scale(theta, df, comment_map)

    # =====================================================
    # 🔥 FIX: PICK A VALID COMMENT FROM THE SAMPLE
    # =====================================================

    # pick a comment that actually exists in this sample
    valid_original_ids = comment_map

    original_id = valid_original_ids[0]   # or random.choice(valid_original_ids)

    print("\nUSING COMMENT ID:", original_id)

    score = get_comment_score(theta_aligned, comment_map, original_id)

    print("\nCOMMENT SCORE:")
    print(score)

    true = df[df["comment_id"] == original_id]["hate_speech_score"].mean()

    print("\nHF SCORE:", true)
    print("DIFF:", score - true)
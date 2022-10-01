import pandas as pd
import torch


def analyze_latent_space(source_latent_space: torch.Tensor, target_latent_space: torch.Tensor):
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Calculando A-distance.")
    a_distance = calcular_a_distance(source_latent_space, target_latent_space)
    logger.info(f"A-distance: {a_distance:.4f}")

    logger.info("Aplicando UMAP.")
    df_features = aplicar_umap(source_latent_space, target_latent_space)
    logger.info("UMAP aplicado.")

    return a_distance, df_features


def aplicar_umap(source_latent_space, target_latent_space):
    import umap

    mapper = umap.UMAP(random_state=33)
    source_feature = mapper.fit_transform(source_latent_space)
    target_feature = mapper.transform(target_latent_space)

    df_mnist = pd.DataFrame({"0": source_feature[:, 0], "1": source_feature[:, 1]})
    df_mnist["label"] = "MNIST"
    df_tds = pd.DataFrame({"0": target_feature[:, 0], "1": target_feature[:, 1]})
    df_tds["label"] = "TDS"
    df_features = pd.concat([df_mnist, df_tds], ignore_index=True)
    return df_features


def calcular_a_distance(source_latent_space, target_latent_space):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    df_source = pd.DataFrame(source_latent_space.numpy())
    df_source["y"] = 0
    df_target = pd.DataFrame(target_latent_space.numpy())
    df_target["y"] = 1
    df = pd.concat([df_source, df_target], ignore_index=True)

    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    lr = LogisticRegression()
    lr.fit(df_train.drop(columns="y"), df_train.y)

    acc = accuracy_score(df_test.y, lr.predict(df_test.drop(columns="y")))
    error = 1 - acc
    a_distance = 2 * (1 - 2 * error)

    return a_distance

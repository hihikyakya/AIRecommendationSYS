from pathlib import Path

from .services import A, B, C, D, E, F


def main(dataset:str|Path):
    df_users_data, df_items_metadata=A.solve(dataset)
    B.solve(df_users_data)
    C.solve(df_items_metadata)
    df_product_user_movie = D.solve(df_users_data)
    E.solve(df_users_data)
    F.solve(df_users_data)



if __name__=="__main__":
    main(dataset="./resources/dataset/ml-1m")

    # ------- for debuggings ---------
    # df_users_data, df_items_metadata=A.solve(dataset="./resources/dataset/ml-1m")

    # B.solve(df_users_data)
    # C.solve(df_items_metadata)
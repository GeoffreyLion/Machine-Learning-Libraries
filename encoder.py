# Importing Library
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce

# One Hot Encoding
def one_hot_encoding(df, col):
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    ohc = OneHotEncoder()
    ohe = ohc.fit_transform(df[str(col)].values.reshape(-1, 1)).toarray()
    dfOneHot = pd.DataFrame(ohe, columns = [str(col) + "_" + str(ohc.categories_[0][i]) for i in range(len(ohc.categories_[0]))])
    dfh = pd.concat([df, dfOneHot], axis = 1)
    return dfh

# Ordinal Encoding
# Temp_dict = {'Cold': 1, 'Warm': 2, 'Hot': 3, 'Very Hot': 4}
# df['Temp_Ordinal'] = df.Temperature.map(Temp_dict)

# Helmert Encoding
def helmert_encoding(df, col):
    import pandas as pd
    import category_encoders as ce
    encoder = ce.HelmertEncoder(cols = col, drop_invariant=True)
    dfh = encoder.fit_transform(df[col])
    df = pd.concat([df, dfh], axis = 1)
    return df

# Frequency Encoding
def frequency_encoding(df, col):
    fe = df.groupby(col).size()/len(df)
    df.loc[:, str(col) + "_freq_encode"] = df[col].map(fe)
    return df

# Mean Encoding
def mean_encoding(df, col, target):
    mean = df[target].mean()
    agg = df.groupby(col)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    weight = 100
    # Smoothing
    smooth = (counts*means + weight*mean)/(counts+weight)
    # Replace by the accorded smoothed mean
    df.loc[:, str(col) + 'smean_enc'] = df[col].map(smooth)
    return df


import matplotlib.pyplot as plt
import seaborn as sns

def univariate_analysis(df, column, plot_type='hist'):
    if plot_type == 'hist':
        sns.histplot(df[column])
    elif plot_type == 'box':
        sns.boxplot(x=df[column])
    plt.title(f"Univariate Analysis: {column}")
    plt.show()

def bivariate_analysis(df, x_col, y_col='class'):
    sns.barplot(x=df[x_col], y=df[y_col])
    plt.title(f"{x_col} vs. {y_col}")
    plt.show()
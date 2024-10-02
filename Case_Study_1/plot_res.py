import pandas as pd
import matplotlib.pyplot as plt


# Create a scatter plot
def plot_scatter(df):
    fig, ax = plt.subplots()

    # Group by the configuration column
    for config, group in df.groupby('config'):
        ax.scatter(group['time/iter'], group['Num Constraints'], label=config)
        
    ax.set_xlabel('Time (wu) per iteration')
    ax.set_ylabel('Number of Constraints')
    ax.set_title('Time per iteration vs Number of Constraints for Different Configurations')
    ax.legend(title='Configuration')
    plt.show()

    fig, ax = plt.subplots()

    # # Group by the configuration column
    # for config, group in df.groupby('config'):
    #     ax.scatter(group['Run Time (wu)'], group['Num Iters'], label=config)
        
    # ax.set_xlabel('Solve Time (wu)')
    # ax.set_ylabel('Num Iters')
    # ax.set_title('Solve Time vs Num Iters for Different Configurations')
    # ax.legend(title='Configuration')
    # plt.show()


df = pd.read_csv(r".\data\track_k_e-Sheet3.csv")
print(df.head())
plot_scatter(df)
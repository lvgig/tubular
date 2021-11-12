def one_way_summary_plot(df, column, response="y"):
    """Function to produce a rough one-way summary plot of a specific column.

    Specifically plot averge response (right y axis) and number of records (left
    y axis) by the selected column (x axis).

    """

    agg = df.groupby(column).agg({column: ["count"], "y": ["mean"]})

    ax = agg.plot.bar(y=(column, "count"), ylabel="count", figsize=(8, 5))

    agg.plot(
        y=("y", "mean"),
        style=":",
        marker=".",
        c="k",
        ax=ax,
        use_index=False,
        secondary_y=True,
        mark_right=False,
        rot=90,
        title=f"one-way summary of {column}",
    )

    ax.right_ax.set_ylabel("mean y")

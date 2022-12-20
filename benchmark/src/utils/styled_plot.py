from distutils.spawn import find_executable

import matplotlib

matplotlib.use("Agg")
import base64
import io

import matplotlib.pyplot as plt

# IEEETrans double column standard
FIG_WIDTH = 252.0 / 72.27  # 1pt is 1/72.27 inches
FIG_HEIGHT = FIG_WIDTH / 1.618  # golden ratio


class StyledPlot:
    """
    Overwrites default settings from matplotlib.pyplot.
    If a function is not overwritten, the default function will be used.
    """

    def __init__(self):
        plt.style.use("seaborn")

        # Set MatPlotLib defaults
        if find_executable("latex"):
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern"],
                }
            )

        plt.rc("figure", autolayout=True)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("axes", labelsize=12)
        plt.rc("axes", titlesize=12)
        plt.rc("legend", fontsize=8)

        self.plt = plt

    def figure(self, cols=1, rows=1, dpi=200):
        # Clean all
        self.plt.cla()
        self.plt.clf()

        f = self.plt.figure(figsize=(FIG_WIDTH * cols, FIG_HEIGHT * rows), dpi=dpi)
        f.tight_layout()

        return f

    def save_figure(self, filename):
        self.plt.savefig(filename, dpi=400, bbox_inches="tight")
        self.plt.close()

    def render(self):
        # Ccreate a virtual file which matplotlib can use to save the figure
        buffer = io.BytesIO()
        self.plt.savefig(buffer, dpi=400, bbox_inches="tight")
        buffer.seek(0)

        # Display any kind of image taken from
        # https://github.com/plotly/dash/issues/71
        encoded_image = base64.b64encode(buffer.read())
        return "data:image/png;base64,{}".format(encoded_image.decode())

    def xlim(self, xmin, xmax):
        xmin_with_margin = xmin - 0.05 * (xmax - xmin)
        xmax_with_margin = xmax + 0.05 * (xmax - xmin)
        self.plt.xlim(xmin_with_margin, xmax_with_margin)

    def ylim(self, ymin, ymax, margin=True):
        if margin:
            ymin_with_margin = ymin - 0.05 * (ymax - ymin)
            ymax_with_margin = ymax + 0.05 * (ymax - ymin)
            self.plt.ylim(ymin_with_margin, ymax_with_margin)
        else:
            self.plt.ylim(ymin, ymax)

    # def grid(self):
    #    pass
    #    #self.plt.grid(b=True, color='black', linestyle='--', linewidth=0.5, axis='y', zorder=0, alpha=0.5)

    def boxplot(self, values, positions, color, widths=0.5):
        bp = self.plt.boxplot(values, positions=positions, patch_artist=True, widths=widths)

        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set(linewidth=0.3)

        for whisker in bp["whiskers"]:
            whisker.set(color=color, linewidth=0.5)

        for cap in bp["caps"]:
            cap.set(color=color, linewidth=0.5)

        for median in bp["medians"]:
            median.set(color="black", linewidth=0.5)

        for flier in bp["fliers"]:
            flier.set(
                marker="o",
                markersize=3,
                markerfacecolor=color,
                linestyle="none",
                markeredgecolor="none",
                color=color,
                alpha=0.5,
            )

    def legend(self, cols=1, loc=None, title=None, outside=False):
        kwargs = {
            "ncol": cols,
            "columnspacing": 0.8,
            "labelspacing": 0,
            "loc": loc,
            "fancybox": False,
            "framealpha": 0.8,
            "frameon": True,
            "borderaxespad": 0.4,
            "facecolor": "white",
            "title": title,
        }

        if loc is not None:
            kwargs["loc"] = loc

        if outside:
            kwargs.update({"loc": "upper left", "bbox_to_anchor": (1, 1)})

        legend = self.plt.legend(**kwargs)
        legend.set_zorder(500)

        if outside:
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_edgecolor("white")
        else:
            legend.get_frame().set_linewidth(0.5)
            legend.get_frame().set_edgecolor("gray")

    def get_color(self, id):
        import seaborn as sns

        pal = sns.color_palette()
        hex_codes = pal.as_hex()

        return hex_codes[id % len(hex_codes)]

    def __getattr__(self, name):
        """
        Make sure we access self.plt directly.
        """

        try:
            return self.__getattribute__(name)
        except:
            return self.plt.__getattribute__(name)


plt = StyledPlot()

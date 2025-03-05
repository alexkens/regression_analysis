from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt


# salary data: https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression/data
DATA = "salary_dataset.csv"
TRIVIAL = "trivial_case.csv"
PATH = DATA

def simple_linear_regression():
    df = pd.read_csv(PATH)

    mu_x = df["YearsExperience"].mean()
    mu_y = df["Salary"].mean()
    x = df["YearsExperience"].values
    y = df["Salary"].values

    numerator = sum([(xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)])
    denominator = sum([(xi - mu_x) ** 2 for xi in x])
    beta = numerator / denominator
    alpha = mu_y - beta * mu_x

    U = [round(xi * beta + alpha, 2) for xi in x]
    print(U)

    return alpha, beta


def plot_residuals_by_predicted(alpha, beta):
    df = pd.read_csv(PATH)
    x = df["YearsExperience"]
    y = df["Salary"]
    residuals = [abs(alpha + beta * xi - yi) for xi, yi in zip(x, y)]

    plt.scatter(y, residuals)
    plt.axhline(mean(residuals), color='red', linestyle="--")

    # style
    plt.title("Residuals by Predicted")
    plt.xlabel("Salary")
    plt.ylabel("Residuals")

    plt.show()


def plot(alpha, beta):
    df = pd.read_csv(PATH)

    # data
    plt.scatter(df["YearsExperience"], df["Salary"])
    
    # line
    x1 = df["YearsExperience"].values[0]
    x2 = df["YearsExperience"].values[-1]
    y1 = alpha + beta * x1
    y2 = alpha + beta * x2
    plt.plot([x1, x2], [y1, y2], c="g")

    # residuals
    x = df["YearsExperience"]
    y = df["Salary"]
    f = [alpha + beta * xi for xi in x]
    residuals = [abs(alpha + beta * xi - yi) for xi, yi in zip(x, y)]
    plt.scatter(x, y, color="#1f77b4")

    for xi, yi, ri, fi in zip(x, y, residuals, f):
        if fi > yi:
            ymin = yi
            ymax = fi
        else: # fi < yi
            ymin = fi
            ymax = yi
        plt.vlines(x=xi, ymin=ymin, ymax=ymax, colors='red', linestyles='--', lw=2)
        # plt.text(xi + 0.1, ((ymax - ymin) / 2) + ymin, s=f"{ri:.1f}", horizontalalignment='left')

    # style
    plt.title("Simple Linear Regression")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.savefig("residuals.png")
    #plt.show()


alpha, beta = simple_linear_regression()
print(alpha, beta)
plot(alpha, beta)

"""df = pd.read_csv("perfect_line.csv")
plt.scatter(df["YearsExperience"], df["Salary"])
plt.title("Simple Linear Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.savefig("perfect_line.png")"""

#plot(alpha, beta)
#plot_residuals_by_predicted(alpha, beta)

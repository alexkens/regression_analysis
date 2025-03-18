from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt


# salary data: https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression/data
# insurance data: https://www.kaggle.com/datasets/mirichoi0218/insurance
# ad data: https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression

DATA_SALARY = "data/salary_dataset.csv"
DATA_INSURANCE = "data/insurance.csv"
DATA_ADS = "data/advertising.csv"
TRIVIAL = "data/trivial_case.csv"


def simple_linear_regression(path, ind_var, dep_var):
    df = pd.read_csv(path)

    mu_x = df[ind_var].mean()
    mu_y = df[dep_var].mean()
    x = df[ind_var].values
    y = df[dep_var].values

    numerator = sum([(xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)])
    denominator = sum([(xi - mu_x) ** 2 for xi in x])
    beta = numerator / denominator
    alpha = mu_y - beta * mu_x

    return alpha, beta


def multiple_linear_regression(path, ind_vars, dep_var):
    pass


def plot_residuals_by_predicted(alpha, beta, path):
    df = pd.read_csv(path)
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


def plot(path, ind_var, dep_var, residual_flag=False, save_flag=False):
    # data
    alpha, beta = simple_linear_regression(path, ind_var, dep_var)
    df = pd.read_csv(path)
    plt.scatter(df[ind_var], df[dep_var])
    
    # line
    sorted_list = sorted(df[ind_var].values)
    x1 = sorted_list[0]
    x2 = sorted_list[-1]
    y1 = alpha + beta * x1
    y2 = alpha + beta * x2
    plt.plot([x1, x2], [y1, y2], c="g")

    # residuals
    if residual_flag:
        x = df[ind_var]
        y = df[dep_var]
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
    plt.xlabel(ind_var)
    plt.ylabel(dep_var)
    if save_flag:
        s = path
        print(s)
        if ".csv" in s:
            s = s.replace("data/", "").replace(".csv", "")
            print(s)
        #plt.savefig(s, format="png")
        plt.savefig(f"{s}.png")
    else:
        plt.show()
    plt.close()
    


def get_squared_error(path, ind_var, dep_var):
    alpha, beta = simple_linear_regression(path, ind_var, dep_var)
    df = pd.read_csv(path)

    X = df[ind_var].values
    Y = df[dep_var].values

    U = [((xi * beta + alpha) - yi) ** 2 for xi, yi in zip(X, Y)]
    return sum(U)


error1 = get_squared_error(DATA_SALARY, "YearsExperience", "Salary")
error2 = get_squared_error(DATA_INSURANCE, "age", "charges")
print("{:.2f}, {:.2f}".format(error1, error2))

# plot(DATA_SALARY, "YearsExperience", "Salary", save_flag=True)
# plot(DATA_INSURANCE, "age", "charges", save_flag=False)
# plot(DATA_ADS, "TV", "Sales", save_flag=True)
error3 = get_squared_error(DATA_ADS, "TV", "Sales")
print(error3)

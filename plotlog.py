import matplotlib.pyplot as plt
import re


# print(content)
def plot_log(logfile):
    with open(logfile, 'r+') as f:
        content = f.read()
    pattern = '\(iter:\d+\).*\)'
    number_pattern = '[\d.]+\d+'

    steps = []
    losses = []
    for match in re.finditer(pattern, content):
        match_text = match.group()
        step, loss, second = re.findall(number_pattern, match_text)
        steps.append(int(step) // 1000)
        losses.append(float(loss))
    # print(losses)
    print(f'ploting {logfile}...')
    plt.plot(steps, losses)


if __name__ == '__main__':
    file1 = 'wavenet-baseline-36layer-1022.log'
    file2 = 'train1029_cp.log'
    # file3 = 'pulse_repeat1/wavenet-stonefree-12layer-1027.log'
    plot_log(file1)
    # plt.show()
    plot_log(file2)
    # plt.show()
    # plot_log(file3)
    plt.show()

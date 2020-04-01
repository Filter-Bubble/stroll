import torch

from stroll.labels import frame_codec, role_codec

from sklearn.metrics import confusion_matrix, classification_report

from progress.bar import Bar
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate(net, evalloader, fig_name, batch_size=50):
    predicted_frames = []
    gold_frames = []

    predicted_roles1 = []
    predicted_roles2 = []
    predicted_roles3 = []
    gold_roles = []

    progbar = Bar('Evaluating', max=len(evalloader))

    net.eval()
    with torch.no_grad():
        for g in evalloader:
            lf, lr = net(g)
            lowest = torch.min(lr)

            # Get best frame
            _, pf = torch.max(lf, dim=1)

            # Get best role, and set its score to zero
            _, pr1st = torch.max(lr, dim=1)
            for i in range(len(g)):
                lr[i, pr1st[i]] = lowest.item()

            # Get second best role, and set its score to zero
            _, pr2nd = torch.max(lr, dim=1)
            for i in range(len(g)):
                lr[i, pr2nd[i]] = lowest.item()

            # Get third best role
            _, pr3rd = torch.max(lr, dim=1)

            gf = g.ndata['frame']
            gr = g.ndata['role']

            predicted_frames += frame_codec.inverse_transform(pf).tolist()
            gold_frames += frame_codec.inverse_transform(gf).tolist()

            predicted_roles1 += role_codec.inverse_transform(pr1st).tolist()
            predicted_roles2 += role_codec.inverse_transform(pr2nd).tolist()
            predicted_roles3 += role_codec.inverse_transform(pr3rd).tolist()
            gold_roles += role_codec.inverse_transform(gr).tolist()

            progbar.next(batch_size)

    progbar.finish()

    main_args = ['Arg0', 'Arg1', 'Arg2', 'Arg3', 'Arg4', 'Arg5']
    reduced_gold = []
    for label in gold_roles:
        if label in main_args:
            reduced_gold.append('Arg')
        elif label == '_':
            reduced_gold.append('_')
        else:
            reduced_gold.append('Mod')

    reduced_pred = []
    for label in predicted_roles1:
        if label in main_args:
            reduced_pred.append('Arg')
        elif label == '_':
            reduced_pred.append('_')
        else:
            reduced_pred.append('Mod')

    norm = None
    labels = role_codec.classes_
    conf_frames = confusion_matrix(gold_frames, predicted_frames,
                                   normalize=norm)
    conf_roles1 = confusion_matrix(gold_roles, predicted_roles1, labels=labels,
                                   normalize=norm)
    conf_roles2 = confusion_matrix(gold_roles, predicted_roles2, labels=labels,
                                   normalize=norm)
    conf_roles3 = confusion_matrix(gold_roles, predicted_roles3, labels=labels,
                                   normalize=norm)
    conf_reduced = confusion_matrix(reduced_gold, reduced_pred,
                                    normalize=norm)

    print(classification_report(gold_frames, predicted_frames))
    print(classification_report(gold_roles, predicted_roles1))
    print(classification_report(reduced_gold, reduced_pred))

    print('\n -- \n')

    print('Frames')
    print(conf_frames)
    print('Roles - best')
    print(conf_roles1)
    print('\n')
    print('Roles - second')
    print(conf_roles2)
    print('\n')
    print('Roles - third')
    print(conf_roles3)
    print('\n')
    print('Roles - simplified')
    print(conf_reduced)

    # Calculate the normalized confusion matrix
    norm = 'true'
    conf_frames = confusion_matrix(gold_frames, predicted_frames,
                                   normalize=norm)
    conf_roles1 = confusion_matrix(gold_roles, predicted_roles1, labels=labels,
                                   normalize=norm)
    conf_roles2 = confusion_matrix(gold_roles, predicted_roles2, labels=labels,
                                   normalize=norm)
    conf_roles3 = confusion_matrix(gold_roles, predicted_roles3, labels=labels,
                                   normalize=norm)
    conf_reduced = confusion_matrix(reduced_gold, reduced_pred,
                                    normalize=norm)

    # take the model fileanme (without pt) as figure name
    fmt = "3.0f"

    plt.figure(figsize=[10., 10.])
    sns.heatmap(
            100. * conf_roles1, fmt=fmt, annot=True, cbar=False,
            cmap="Greens", xticklabels=labels, yticklabels=labels
            )
    plt.savefig(fig_name + 'roles1.png')

    plt.figure(figsize=[10., 10.])
    sns.heatmap(
            100. * conf_roles2, fmt=fmt, annot=True, cbar=False,
            cmap="Greens", xticklabels=labels, yticklabels=labels
            )
    plt.savefig(fig_name + 'roles2.png')

    plt.figure(figsize=[10., 10.])
    sns.heatmap(
            100. * conf_roles3, fmt=fmt, annot=True, cbar=False,
            cmap="Greens", xticklabels=labels, yticklabels=labels
            )
    plt.savefig(fig_name + 'roles3.png')

    plt.figure(figsize=[10., 10.])
    sns.heatmap(
            100. * conf_reduced,
            fmt=fmt, annot=True, cbar=False, cmap="Greens",
            xticklabels=['Arg', 'ArgM', '_'], yticklabels=['Arg', 'Mod', '_']
            )
    plt.savefig(fig_name + 'roles_red.png')

    plt.figure(figsize=[10., 10.])
    labels = frame_codec.classes_
    sns.heatmap(
            100. * conf_frames, fmt=fmt, annot=True, cbar=False, cmap="Greens",
            xticklabels=labels, yticklabels=labels
            )
    plt.savefig(fig_name + 'frames.png')

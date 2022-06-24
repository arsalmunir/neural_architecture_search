import torchvision
import torchvision.transforms as transforms


def get_dataset_train(**kwargs):
    # Return train dataset
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset_root = kwargs['root']

    return torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=train_transform)


def get_dataset_test(**kwargs):
    # Return test dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset_root = kwargs['root']

    return torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=test_transform)


if __name__ == '__main__':
    train_set = get_dataset_train(root='data/cifar10')
    test_set = get_dataset_test(root='data/cifar10')

    print('\n--- CIFAR10 ---')
    print('Train set length = {}\nTest set length = {}'.format(train_set.__len__(), test_set.__len__()))

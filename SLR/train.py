import os
import torch
import argparse

from twostream.s3d_twostream import TwoStreamS3D
from dataset.dataloader import VideoDataSet
from utils import EarlyStopping, trainning_phase, testing_phase, count_parameters

parser = argparse.ArgumentParser(description="Setting Path")

parser.add_argument("--root_path", type=str, help="Enter your data folder path", default=None)
parser.add_argument("--pretrained", type=str, help="Enter your pretrained path", default=None)
parser.add_argument("--model_save", type=str, help="Enter your save path", default=None)

parser.add_argument("--data_name", type=str, help="Enter name of dataset", default=None)
parser.add_argument("--num_frames", type=int, help="Enter number frame of video", default=16)
parser.add_argument("--image_size", type=int, help="Enter image size of frame", default=224)

parser.add_argument("--num_classes", type=int, help="Enter number of classification", default=226)
parser.add_argument("--num_epochs", type=int, help="Enter number of epoch for training", default=20)
parser.add_argument("--batch_size", type=int, help="Enter batch size of a iteration", default=8)
parser.add_argument("--lr", type=float, help="Enter learning rate for trainning", default=1e-3)
parser.add_argument("--max_viz", type=int, help="Enter max_viz for validation phase", default=2)

args = parser.parse_args()



if __name__ == "__main__":

    DEVICE = None

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    train_loader = torch.utils.data.DataLoader(VideoDataSet(folder_root=args.root_path, num_frames=args.num_frames, 
                                                            data_name=args.data_name, split="train", image_size=args.image_size), 
                                                            batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    test_loader = torch.utils.data.DataLoader(VideoDataSet(folder_root=args.root_path, num_frames=args.num_frames, 
                                                           data_name=args.data_name, split="test", image_size=args.image_size), 
                                                           batch_size=4, shuffle=True, num_workers=4)
    
    
    print(f"Train on device: {DEVICE}")
    print("\n")
    print(f"Train on dataset: {args.data_name} dataset")
    print(f"Samples in train datase: {len(train_loader) * args.batch_size}")
    print(f"Samples in test datase: {len(test_loader) * 4}")
    print("\n")

    save_path = args.model_save
    model = TwoStreamS3D(num_classes=226)
    if args.pretrained != None:
        print("Load Pretrain Weight !!")
        print("\n")
        model.load_state_dict(torch.load(args.pretrained))
    if args.num_classes != 226:
        model.classifier[1] = torch.nn.Conv3d(1024, args.num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    
    print("Model Detail")
    total_parameters = count_parameters(model)
    print(f"Total parameters in the model is: {round(total_parameters / 1000000, 4)}M parameters")
    print("\n")

    model = model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=save_path)

    print(f"Number of epochs: {args.num_epochs}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("\n")

    for epoch in range(1, args.num_epochs + 1):

        # print(f"EPOCH [{epoch}/{args.num_epochs}]: ")
        # print("Trainning Phase")
        # epoch_train_loss, epoch__train_acc = trainning_phase(model, train_loader, optimizer, scheduler, criterion, DEVICE)
        # print(f"Epoch Loss Train: {round(epoch_train_loss, 4)} - Epochs Accuracy Train: {round(epoch__train_acc, 4)}")

        if epoch % args.max_viz == 0:

            print("Validation Phase")
            epoch_test_loss, epoch_test_acc = testing_phase(model, test_loader, criterion, DEVICE)
            print(f"Epoch Loss Validation: {round(epoch_test_loss, 4)} - Epochs Accuracy Test: {round(epoch_test_acc, 4)}")
            early_stopping(epoch_test_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        print("\n")

    print("Testing Phase")
    epoch_test_loss, epoch_test_acc = testing_phase(model, test_loader, criterion, DEVICE)
    print(f"Epoch Loss Test: {round(epoch_test_loss, 4)} - Epochs Accuracy Test: {round(epoch_test_acc, 4)}")
    torch.save(model.state_dict(), save_path)
    print("Finished Training Model !!")





    
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from data.USCAnnot16Loader import USCAnnot16Dataset, USCTIMITFrameDataset
from src.models.contrastive_model import AudioVisionContrastiveModel
from src.losses.loss_factory import BuildLoss


NUM_CLASSES = {
    "": 6 * 8 * 3,  # manner-place-voicing 的组合分类
    "manner": 6,
    "place": 8,
    "voicing": 3,
}


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_cos_loss = 0.0

    for batch in tqdm(dataloader):
        image = batch["image"].to(device)
        audio = batch["audio"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(image=image, audio=audio)

        losses = criterion(
            logits=outputs["logits"],
            labels=label,
            visual_flat=outputs["visual_flat"],
            audio_flat=outputs["audio_flat"],
        )

        losses["loss"].backward()
        optimizer.step()

        running_loss += losses["loss"].item()
        running_cls_loss += losses["cls_loss"].item()
        running_cos_loss += losses["contrast_loss"].item()

    n = len(dataloader)

    return {
        "loss": running_loss / n,
        "cls_loss": running_cls_loss / n,
        "contrast_loss": running_cos_loss / n,
    }


def main():
    task = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"


    train_dataset = USCAnnot16Dataset(
    data_root="data",
    image_dataset_dir="USC-annot-16",
    label_dataset_dir="processed/labels_annot_16",
    subjects=["sub009"],
    tasks=["bvt"],
    image_size=128,
    target_sample_rate=16000,
    train=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = AudioVisionContrastiveModel(
        num_classes=NUM_CLASSES[task],
        visual_tokens=65,
        target_tokens=31,
        hidden_size=768,
        lambda_cosine=0.1,
        task=task,
    ).to(device)

    criterion = BuildLoss(lambda_contrast=0.1)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=5e-4,
    )

    for epoch in range(30):
        train_log = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        print(f"Epoch {epoch + 1}: {train_log}")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"checkpoint_epoch_{epoch + 1}.pt",
        )


if __name__ == "__main__":
    main()
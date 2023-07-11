import os
import torch
import shutil
import warnings
import numpy as np
from torch.optim import Adam
import json
from utils.io_utils import check_dir
from gendata import get_dataloader
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import balanced_accuracy_score, f1_score


class TrainModel(object):
    def __init__(
        self,
        model,
        dataset,
        device,
        graph_classification=True,
        save_dir=None,
        save_name="model",
        **kwargs,
    ):
        self.model = model
        self.dataset = dataset  # train_mask, eval_mask, test_mask
        self.loader = None
        self.device = device
        self.graph_classification = graph_classification
        self.node_classification = not graph_classification

        self.optimizer = None
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        check_dir(self.save_dir)

        if self.graph_classification:
            dataloader_params = kwargs.get("dataloader_params")
            self.loader, _, _, _ = get_dataloader(dataset, **dataloader_params)

    def __loss__(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def _train_batch(self, data, labels):
        logits = self.model(data=data)
        if self.graph_classification:
            loss = self.__loss__(logits, labels)
        else:
            loss = self.__loss__(logits[data.train_mask], labels[data.train_mask])

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
        self.optimizer.step()
        return loss.item()

    def _eval_batch(self, data, labels, **kwargs):
        self.model.eval()
        logits = self.model(data)
        if self.graph_classification:
            loss = self.__loss__(logits, labels)
        else:
            mask = kwargs.get("mask")
            if mask is None:
                warnings.warn("The node mask is None")
                mask = torch.ones(labels.shape[0])
            loss = self.__loss__(logits[mask], labels[mask])
        loss = loss.item()
        preds = logits.argmax(-1)
        return loss, preds

    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        if self.graph_classification:
            losses, accs, balanced_accs, f1_scores = [], [], [], []
            for batch in self.loader["eval"]:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                batch_preds, batch.y = batch_preds.cpu(), batch.y.cpu()
                accs.append(batch_preds == batch.y)
                balanced_accs.append(balanced_accuracy_score(batch.y, batch_preds))
                f1_scores.append(f1_score(batch.y, batch_preds, average="weighted"))
            eval_loss = torch.tensor(losses).mean().item()
            eval_acc = torch.cat(accs, dim=-1).float().mean().item()
            eval_balanced_acc = np.mean(balanced_accs)
            eval_f1_score = np.mean(f1_scores)
        else:
            data = self.dataset.data.to(self.device)
            eval_loss, preds = self._eval_batch(data, data.y, mask=data.val_mask)
            preds, data.y = preds.cpu(), data.y.cpu()
            eval_acc = (preds == data.y).float().mean().item()
            eval_balanced_acc = balanced_accuracy_score(data.y, preds)
            eval_f1_score = f1_score(data.y, preds, average="weighted")
        return eval_loss, eval_acc, eval_balanced_acc, eval_f1_score

    def test(self):
        state_dict = torch.load(
            os.path.join(self.save_dir, f"{self.save_name}_best.pth")
        )["net"]
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.graph_classification:
            losses, preds, accs, balanced_accs, f1_scores = [], [], [], [], []
            for batch in self.loader["test"]:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                preds.append(batch_preds)
                batch_preds, batch.y = batch_preds.cpu(), batch.y.cpu()
                accs.append(batch_preds == batch.y)
                balanced_accs.append(balanced_accuracy_score(batch.y, batch_preds))
                f1_scores.append(f1_score(batch.y, batch_preds, average="weighted"))
            test_loss = torch.tensor(losses).mean().item()
            preds = torch.cat(preds, dim=-1)
            test_acc = torch.cat(accs, dim=-1).float().mean().item()
            test_balanced_acc = np.mean(balanced_accs)
            test_f1_score = np.mean(f1_scores)
        else:
            data = self.dataset.data.to(self.device)
            test_loss, preds = self._eval_batch(data, data.y, mask=data.test_mask)
            preds, y = preds.cpu(), data.y.cpu()
            test_acc = (preds == data.y).float().mean().item()
            test_balanced_acc = balanced_accuracy_score(data.y, preds)
            test_f1_score = f1_score(data.y, preds, average="weighted")
        print(
            f"Test loss: {test_loss:.4f}, test acc {test_acc:.4f}, balanced test acc {test_balanced_acc:.4f}, test f1 score {test_f1_score:.4f}"
        )
        scores = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_balanced_acc": test_balanced_acc,
            "test_f1_score": test_f1_score,
        }
        # self.save_scores(scores)
        return scores, preds

    def train(self, train_params=None, optimizer_params=None):
        num_epochs = train_params["num_epochs"]
        num_early_stop = train_params["num_early_stop"]
        milestones = train_params["milestones"]
        gamma = train_params["gamma"]

        if optimizer_params is None:
            self.optimizer = Adam(self.model.parameters())
        else:
            self.optimizer = Adam(self.model.parameters(), **optimizer_params)

        if milestones is not None and gamma is not None:
            lr_schedule = MultiStepLR(
                self.optimizer, milestones=milestones, gamma=gamma
            )
        else:
            lr_schedule = None

        self.model.to(self.device)
        best_eval_acc = 0.0
        best_eval_loss = 0.0
        early_stop_counter = 0
        for epoch in range(num_epochs):
            is_best = False
            self.model.train()
            if self.graph_classification:
                losses = []
                for batch in self.loader["train"]:
                    batch = batch.to(self.device)
                    loss = self._train_batch(batch, batch.y)
                    losses.append(loss)
                train_loss = torch.FloatTensor(losses).mean().item()

            else:
                data = self.dataset.data.to(self.device)
                train_loss = self._train_batch(data, data.y)

            with torch.no_grad():
                eval_loss, eval_acc, eval_balanced_acc, eval_f1_score = self.eval()
            print(
                f"Epoch:{epoch}, Training_loss:{train_loss:.4f}, Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}, Eval_balanced_acc:{eval_balanced_acc:.4f}, Eval_f1_score:{eval_f1_score:.4f}"
            )
            if num_early_stop > 0:
                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                    break
            if lr_schedule:
                lr_schedule.step()

            if best_eval_acc < eval_acc:
                is_best = True
                best_eval_acc = eval_acc
            recording = {"epoch": epoch, "is_best": str(is_best)}
            if self.save:
                self.save_model(is_best, recording=recording)

    def save_model(self, is_best=False, recording=None):
        self.model.to("cpu")
        state = {"net": self.model.state_dict()}
        for key, value in recording.items():
            state[key] = value
        latest_pth_name = f"{self.save_name}_latest.pth"
        best_pth_name = f"{self.save_name}_best.pth"
        ckpt_path = os.path.join(self.save_dir, latest_pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            print("saving best...")
            shutil.copy(ckpt_path, os.path.join(self.save_dir, best_pth_name))
        self.model.to(self.device)

    def load_model(self):
        state_dict = torch.load(
            os.path.join(self.save_dir, f"{self.save_name}_best.pth")
        )["net"]
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def save_scores(self, scores):
        with open(os.path.join(self.save_dir, f"{self.save_name}_scores.json"), "w") as f:
            json.dump(scores, f)


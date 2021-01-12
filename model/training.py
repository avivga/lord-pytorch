import os
import itertools
import pickle
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.modules import LatentModel, AmortizedModel, VGGDistance
from model.utils import AverageMeter, NamedTensorDataset


class Lord:

	def __init__(self, config=None):
		super().__init__()

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.latent_model = None
		self.amortized_model = None

	def load(self, model_dir, latent=True, amortized=True):
		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			self.config = pickle.load(config_fd)

		if latent:
			self.latent_model = LatentModel(self.config)
			self.latent_model.load_state_dict(torch.load(os.path.join(model_dir, 'latent.pth')))

		if amortized:
			self.amortized_model = AmortizedModel(self.config)
			self.amortized_model.load_state_dict(torch.load(os.path.join(model_dir, 'amortized.pth')))

	def save(self, model_dir, latent=True, amortized=True):
		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		if latent:
			torch.save(self.latent_model.state_dict(), os.path.join(model_dir, 'latent.pth'))

		if amortized:
			torch.save(self.amortized_model.state_dict(), os.path.join(model_dir, 'amortized.pth'))

	def train_latent(self, imgs, classes, model_dir, tensorboard_dir):
		self.latent_model = LatentModel(self.config)

		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['train']['batch_size'],
			shuffle=True, sampler=None, batch_sampler=None,
			num_workers=1, pin_memory=True, drop_last=True
		)

		self.latent_model.init()
		self.latent_model.to(self.device)

		criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)

		optimizer = Adam([
			{
				'params': itertools.chain(self.latent_model.modulation.parameters(), self.latent_model.generator.parameters()),
				'lr': self.config['train']['learning_rate']['generator']
			},
			{
				'params': itertools.chain(self.latent_model.content_embedding.parameters(), self.latent_model.class_embedding.parameters()),
				'lr': self.config['train']['learning_rate']['latent']
			}
		], betas=(0.5, 0.999))

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['train']['n_epochs'] * len(data_loader),
			eta_min=self.config['train']['learning_rate']['min']
		)

		summary = SummaryWriter(log_dir=tensorboard_dir)

		train_loss = AverageMeter()
		for epoch in range(self.config['train']['n_epochs']):
			self.latent_model.train()
			train_loss.reset()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				optimizer.zero_grad()
				out = self.latent_model(batch['img_id'], batch['class_id'])

				content_penalty = torch.sum(out['content_code'] ** 2, dim=1).mean()
				loss = criterion(out['img'], batch['img']) + self.config['content_decay'] * content_penalty

				loss.backward()
				optimizer.step()
				scheduler.step()

				train_loss.update(loss.item())
				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(loss=train_loss.avg)

			pbar.close()
			self.save(model_dir, latent=True, amortized=False)

			summary.add_scalar(tag='loss', scalar_value=train_loss.avg, global_step=epoch)

			fixed_sample_img = self.generate_samples(dataset, randomized=False)
			random_sample_img = self.generate_samples(dataset, randomized=True)

			summary.add_image(tag='sample-fixed', img_tensor=fixed_sample_img, global_step=epoch)
			summary.add_image(tag='sample-random', img_tensor=random_sample_img, global_step=epoch)

		summary.close()

	def train_amortized(self, imgs, classes, model_dir, tensorboard_dir):
		self.amortized_model = AmortizedModel(self.config)
		self.amortized_model.modulation.load_state_dict(self.latent_model.modulation.state_dict())
		self.amortized_model.generator.load_state_dict(self.latent_model.generator.state_dict())

		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['train']['batch_size'],
			shuffle=True, sampler=None, batch_sampler=None,
			num_workers=1, pin_memory=True, drop_last=True
		)

		self.latent_model.to(self.device)
		self.amortized_model.to(self.device)

		reconstruction_criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)
		embedding_criterion = nn.MSELoss()

		optimizer = Adam(
			params=self.amortized_model.parameters(),
			lr=self.config['train_encoders']['learning_rate']['max'],
			betas=(0.5, 0.999)
		)

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['train_encoders']['n_epochs'] * len(data_loader),
			eta_min=self.config['train_encoders']['learning_rate']['min']
		)

		summary = SummaryWriter(log_dir=tensorboard_dir)

		train_loss = AverageMeter()
		for epoch in range(self.config['train_encoders']['n_epochs']):
			self.latent_model.eval()
			self.amortized_model.train()

			train_loss.reset()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				optimizer.zero_grad()

				target_content_code = self.latent_model.content_embedding(batch['img_id'])
				target_class_code = self.latent_model.class_embedding(batch['class_id'])

				out = self.amortized_model(batch['img'])

				loss_reconstruction = reconstruction_criterion(out['img'], batch['img'])
				loss_content = embedding_criterion(out['content_code'], target_content_code)
				loss_class = embedding_criterion(out['class_code'], target_class_code)

				loss = loss_reconstruction + 10 * loss_content + 10 * loss_class

				loss.backward()
				optimizer.step()
				scheduler.step()

				train_loss.update(loss.item())
				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(loss=train_loss.avg)

			pbar.close()
			self.save(model_dir, latent=False, amortized=True)

			summary.add_scalar(tag='loss-amortized', scalar_value=loss.item(), global_step=epoch)
			summary.add_scalar(tag='rec-loss-amortized', scalar_value=loss_reconstruction.item(), global_step=epoch)
			summary.add_scalar(tag='content-loss-amortized', scalar_value=loss_content.item(), global_step=epoch)
			summary.add_scalar(tag='class-loss-amortized', scalar_value=loss_class.item(), global_step=epoch)

			fixed_sample_img = self.generate_samples_amortized(dataset, randomized=False)
			random_sample_img = self.generate_samples_amortized(dataset, randomized=True)

			summary.add_image(tag='sample-fixed-amortized', img_tensor=fixed_sample_img, global_step=epoch)
			summary.add_image(tag='sample-random-amortized', img_tensor=random_sample_img, global_step=epoch)

		summary.close()

	def generate_samples(self, dataset, n_samples=5, randomized=False):
		self.latent_model.eval()

		if randomized:
			random = np.random
		else:
			random = np.random.RandomState(seed=1234)

		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		blank = torch.ones_like(samples['img'][0])
		output = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(n_samples):
				out = self.latent_model(samples['img_id'][[j]], samples['class_id'][[i]])
				converted_imgs.append(out['img'][0])

			output.append(torch.cat(converted_imgs, dim=2))

		return torch.cat(output, dim=1)

	def generate_samples_amortized(self, dataset, n_samples=5, randomized=False):
		self.amortized_model.eval()

		if randomized:
			random = np.random
		else:
			random = np.random.RandomState(seed=1234)

		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		blank = torch.ones_like(samples['img'][0])
		output = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(n_samples):
				out = self.amortized_model.convert(samples['img'][[j]], samples['img'][[i]])
				converted_imgs.append(out['img'][0])

			output.append(torch.cat(converted_imgs, dim=2))

		return torch.cat(output, dim=1)

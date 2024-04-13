### Pretraining VAE and get latent features
python Models/vae/vae.py
### Training DDPM using latent features
python main.py --name stock_data_latent_features --config_file Config/stocks_latent.yaml --gpu 0 --train
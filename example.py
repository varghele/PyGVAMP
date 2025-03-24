# Example with MLP Encoder
input_dim = 10
hidden_dims = [64, 32]
output_dim = 5
lag_time = 1

# Create the encoder
mlp_encoder = MLPEncoder(input_dim, hidden_dims, output_dim)

# Create the VAMPNet with the encoder
vampnet_mlp = VAMPNet(encoder=mlp_encoder, lag_time=lag_time)

# Generate some dummy data
data = torch.randn(100, input_dim)  # 100 samples of dimension input_dim
x_t0, x_t1 = vampnet_mlp.create_time_lagged_dataset(data)

# Calculate VAMP score
score = vampnet_mlp.vamp_score(x_t0, x_t1)
print(f"MLP VAMP score: {score.item()}")

# Train
optimizer = torch.optim.Adam(vampnet_mlp.parameters(), lr=0.01)

# Single optimization step
optimizer.zero_grad()
loss = vampnet_mlp.vamp_loss(x_t0, x_t1)
loss.backward()
optimizer.step()

print(f"MLP Loss after one step: {loss.item()}")

# Now try with a CNN encoder
input_channels = 3
hidden_channels = [16, 32, 64]
cnn_encoder = CNNEncoder(input_channels, hidden_channels, output_dim)

# Create the VAMPNet with the CNN encoder
vampnet_cnn = VAMPNet(encoder=cnn_encoder, lag_time=lag_time)

# Different data format for CNN
sequence_length = 20
cnn_data = torch.randn(100, input_channels, sequence_length)
cnn_x_t0, cnn_x_t1 = vampnet_cnn.create_time_lagged_dataset(cnn_data)

# Train CNN-based VAMPNet
cnn_optimizer = torch.optim.Adam(vampnet_cnn.parameters(), lr=0.01)
cnn_loss = vampnet_cnn.vamp_loss(cnn_x_t0, cnn_x_t1)

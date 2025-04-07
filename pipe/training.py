def train_pipeline(vampnet, data_loader, train_config, dataset_info):
    # Set up optimizer
    optimizer = torch.optim.Adam(
        vampnet.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"]
    )

    # Train the model
    losses = vampnet.fit(
        data_loader=data_loader,
        optimizer=optimizer,
        n_epochs=train_config["n_epochs"],
        k=train_config.get("k", None),
        verbose=True
    )

    # Save the model and training information
    run_dir = save_run_info(
        model=vampnet,
        train_config=train_config,
        dataset_info=dataset_info,
        save_dir="./runs"
    )

    # Save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("VAMPNet Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))

    return vampnet, run_dir


# Run training
trained_model, run_dir = train_pipeline(vampnet, data_loader, train_config, dataset_info)

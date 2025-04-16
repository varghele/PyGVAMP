def analyze_pipeline(run_dir, data_loader):
    # Load the model and configurations
    model, train_config, dataset_info = load_run(
        run_dir=run_dir,
        encoder_class=SchNetEncoder,
        vamp_score_class=VAMPScore,
        classifier_class=SoftmaxMLP
    )

    # Put model in evaluation mode
    model.eval()

    # Analyze the model outputs
    results = analyze_vampnet(
        model=model,
        data_loader=data_loader,
        save_folder=os.path.join(run_dir, "analysis"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Generate analysis visualizations
    # ...

    return results


# Analyze the trained model
analysis_results = analyze_pipeline(run_dir, test_loader)

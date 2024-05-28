from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://127.0.0.1:8265")

kick_off_pytorch_benchmark = (
    # Clone the project.
    "git clone -b main git@github.com:javiro/lw_translator || true;"
    # Run the training.
    "python lw_translator/ray_training.py"
    " --data-size-gb=1 --num-epochs=2 --num-workers=1"
)


submission_id = client.submit_job(
    entrypoint=kick_off_pytorch_benchmark,
)

print("Use the following command to follow this Job's logs:")
print(f"ray job logs '{submission_id}' --address http://127.0.0.1:8265 --follow")

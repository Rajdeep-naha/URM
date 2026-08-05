from datasets import load_dataset
dataset = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.1", split="train")
print("Skywork Sample:")
print("chosen:", dataset[0]["chosen"])
print("rejected:", dataset[0]["rejected"])

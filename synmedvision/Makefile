.PHONY: setup data train_stylegan train_diffusion sample eval report demo docker reproduce

setup:
python -m venv .venv && . .venv/bin/activate && pip install -e .

data:
python -m src.cli.prepare --config configs/data.yaml

train_stylegan:
python -m src.cli.train --config configs/train_stylegan.yaml --model stylegan2

train_diffusion:
python -m src.cli.train --config configs/train_diffusion_lora.yaml --model diffusion_lora

sample:
python -m src.cli.generate --n 10000 --out out/samples

eval:
python -m src.cli.evaluate_images --config configs/eval.yaml

report:
python -m src.cli.report --out reports/eval_report.html

demo:
streamlit run src/demo/streamlit_app.py

docker:
docker build -t synmedvision -f docker/Dockerfile .

reproduce: setup data train_stylegan train_diffusion sample eval report

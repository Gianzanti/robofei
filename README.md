# Processo inicial de instalação (Linux Ubuntu 22.04)

1. Instalar Python 3.11

```bash
sudo apt get update
sudo apt install python3.11
```

> IMPORTANTE: a versão Python 3.12 ainda não é suportada corretamente pelo Gymnasium!

2. Instalar pipx

```bash
sudo apt install pipx
```

3. Instalar poetry

```bash
pipx install poetry
```

4. Configurar poetry para guardar localmente os requests

```bash
poetry config virtualenvs.in-project true
```

5. Clonar o repositório: https://github.com/Gianzanti/robofei.git

6. Executar a instalação das dependências

```bash
poetry install
```

7. Caso o Gymnasium ainda esteja na versão v.0.29.1:

editar o arquivo gymnasium/envs/mujoco/mujoco_rendering.py (dentro do .venv/lib/pythonx.xx/site-packages) e substituir `self.data.solver_iter` por `self.data.solver_niter[0]`

8. Ativar o ambiente

```bash
poetry shell
```

Agora é só começar a simular e testar

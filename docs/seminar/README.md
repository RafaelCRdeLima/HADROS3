# Seminário HADROS3

Apresentação web autocontida sobre o pipeline HADROS3, da visualização do sistema Kerr ao transporte local por sítio com GEANT4.

## Apresentar localmente

Abra `index.html` diretamente ou sirva o diretório:

```bash
python3 -m http.server 8000 --directory docs
```

e acesse `http://127.0.0.1:8000/seminar/`.

Controles: setas ou espaço para navegar, `F` para tela cheia, `O` para visão geral, `N` para notas e `?` para ajuda.

## GitHub Pages

O workflow `pages.yml` publica o diretório `docs/` após cada push em `main`. A apresentação fica em:

```text
https://rafaelcrdelima.github.io/HADROS3/seminar/
```

Se o repositório ainda não usa Pages via Actions, selecione **Settings → Pages → Source → GitHub Actions** uma única vez.

## Atualizar snapshots

Os arquivos em `assets/` são cópias congeladas de um run validado. Isso impede que a apresentação dependa do diretório local `output/`, normalmente ignorado pelo Git.

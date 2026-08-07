#!/usr/bin/env python3
"""Baixa os dados públicos do HERA e escreve os subconjuntos usados na apostila.

Uso:  python3 baixar_dados_hera.py

Baixa de https://www.desy.de/h1zeus/herapdf20/ :
  * as seções de choque reduzidas combinadas H1+ZEUS (NC e CC, e+p, 920 GeV);
  * a grade LHAPDF do HERAPDF2.0 LO (28 MB, descartada depois da extração).

Escreve em sandbox/data/hera/ três arquivos pequenos e versionáveis. Só precisa
ser rodado de novo se você quiser mudar as escalas ou a grade de x extraídas.

Referência dos dados:
  H1 and ZEUS Collaborations, Eur. Phys. J. C 75 (2015) 580, arXiv:1506.06042.
"""

import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

BASE = "https://www.desy.de/h1zeus/herapdf20"
AQUI = Path(__file__).resolve().parent
DESTINO = AQUI.parents[0] / "data" / "hera"
CREDITO = [
    "# H1 and ZEUS Collaborations, Eur. Phys. J. C 75 (2015) 580, arXiv:1506.06042",
    f"# fonte: {BASE}/",
    "# gerado por sandbox/apostila/baixar_dados_hera.py",
]


def baixa(url, destino):
    print(f"  baixando {url.rsplit('/', 1)[-1]} ...", flush=True)
    r = subprocess.run(["curl", "-sSL", "-A", "Mozilla/5.0", "--max-time", "600",
                        "-o", str(destino), url], capture_output=True, text=True)
    if r.returncode != 0 or not destino.exists() or destino.stat().st_size == 0:
        raise RuntimeError(f"falha ao baixar {url}: {r.stderr}")
    return destino


# ---------------------------------------------------------------------------
# 1. seções de choque reduzidas
# ---------------------------------------------------------------------------
def extrai_sigma(origem, destino, titulo):
    """Colunas do arquivo do DESY: 0=Q2 1=x 2=y 3=Sigma 4=stat 5=uncor
    6..167=sys1..sys162 168=tot_noproc (incerteza total, em %)."""
    linhas = [l for l in origem.read_text().splitlines() if l.strip()][1:]
    d = np.array([[float(v) for v in l.split()] for l in linhas])
    Q2, x, y, sig, tot = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 168]
    err = sig * tot / 100.0
    ordem = np.lexsort((Q2, x))
    with destino.open("w") as f:
        f.write(f"# {titulo}\n")
        f.write("\n".join(CREDITO) + "\n")
        f.write("# sigma_red = secao de choque reduzida; err = incerteza total"
                " (excluidas as procedurais)\n")
        f.write("# Q2[GeV^2]  x  y  sigma_red  err\n")
        for i in ordem:
            f.write(f"{Q2[i]:.6e} {x[i]:.6e} {y[i]:.6e} {sig[i]:.6e} {err[i]:.6e}\n")
    print(f"  -> {destino.name}: {len(d)} pontos, "
          f"Q2 {Q2.min():.3g}-{Q2.max():.3g} GeV^2, x {x.min():.3g}-{x.max():.3g}")


# ---------------------------------------------------------------------------
# 2. distribuições de pártons
# ---------------------------------------------------------------------------
def le_lhagrid(path):
    """Parser mínimo do formato lhagrid1 do LHAPDF (evita depender do LHAPDF).

    O arquivo é uma sequência de blocos separados por '---'. Cada bloco traz,
    nesta ordem: a grade em x, a grade em Q, os ids de sabor, e depois
    nx*nQ linhas com um valor de xf por sabor, variando Q mais rápido que x.
    """
    blocos = []
    for bruto in path.read_text().split("---")[1:]:
        linhas = [l for l in bruto.splitlines() if l.strip()]
        if len(linhas) < 4:
            continue
        xs = np.array([float(v) for v in linhas[0].split()])
        Qs = np.array([float(v) for v in linhas[1].split()])
        ids = [int(v) for v in linhas[2].split()]
        vals = np.array([[float(v) for v in l.split()] for l in linhas[3:]])
        assert vals.shape == (len(xs) * len(Qs), len(ids)), vals.shape
        blocos.append((xs, Qs, ids, vals.reshape(len(xs), len(Qs), len(ids))))
    return blocos


def faz_xf(blocos):
    def xf(sabor, x_alvo, Q_alvo):
        """xf(x,Q) por interpolação bilinear em (ln x, ln Q)."""
        for xs, Qs, ids, F in blocos:
            if Qs.min() <= Q_alvo <= Qs.max():
                plano = F[:, :, ids.index(sabor)]
                col = np.array([np.interp(np.log(Q_alvo), np.log(Qs), plano[i, :])
                                for i in range(len(xs))])
                return np.interp(np.log(x_alvo), np.log(xs), col)
        raise ValueError(f"Q={Q_alvo} fora dos blocos da grade")
    return xf


def extrai_pdfs(grid, destino, escalas=(10.0, 100.0, 10000.0)):
    xf = faz_xf(le_lhagrid(grid))
    X = np.logspace(-5, np.log10(0.9), 220)
    with destino.open("w") as f:
        f.write("# PDFs do proton: xf(x,Q^2), HERAPDF2.0 LO"
                " (parametrizacao de gluon alternativa)\n")
        f.write("\n".join(CREDITO) + "\n")
        f.write("# extraido do membro central (0000) da grade LHAPDF\n")
        f.write("# Q2[GeV^2]  x  xu_v  xd_v  xubar  xdbar  xs  xc  xb  xg\n")
        for Q2 in escalas:
            Q = np.sqrt(Q2)
            for x in X:
                f.write(" ".join([
                    f"{Q2:.1f}", f"{x:.6e}",
                    f"{xf(2, x, Q) - xf(-2, x, Q):.6e}",
                    f"{xf(1, x, Q) - xf(-1, x, Q):.6e}",
                    f"{xf(-2, x, Q):.6e}", f"{xf(-1, x, Q):.6e}",
                    f"{xf(3, x, Q):.6e}", f"{xf(4, x, Q):.6e}",
                    f"{xf(5, x, Q):.6e}", f"{xf(21, x, Q):.6e}"]) + "\n")
    print(f"  -> {destino.name}: {len(X)} pontos em x, escalas {escalas}")

    # verificação: as regras de contagem de quarks de valência têm de dar 2 e 1
    xg = np.logspace(-7, np.log10(0.999), 4000)
    for Q2 in (escalas[0], escalas[-1]):
        Q = np.sqrt(Q2)
        for sabor, nome, esperado in ((2, "u_v", 2.0), (1, "d_v", 1.0)):
            soma = np.trapezoid(
                np.array([(xf(sabor, x, Q) - xf(-sabor, x, Q)) / x for x in xg]), xg)
            marca = "ok" if abs(soma - esperado) < 5e-3 else "SUSPEITO"
            print(f"     regra de soma  Q2={Q2:>7.0f}  int {nome} dx = "
                  f"{soma:.4f}  (esperado {esperado}) {marca}")


# ---------------------------------------------------------------------------
def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("seções de choque reduzidas:")
        for arq, saida, titulo in [
            ("HERA1+2_NCep_920.dat", "hera_nc_ep_920.dat",
             "Secao de choque reduzida NC e+p, sqrt(s)=318 GeV"),
            ("HERA1+2_CCep.dat", "hera_cc_ep.dat",
             "Secao de choque reduzida CC e+p, sqrt(s)=318 GeV"),
        ]:
            extrai_sigma(baixa(f"{BASE}/{arq}", tmp / arq), DESTINO / saida, titulo)

        print("\ndistribuicoes de partons:")
        tgz = baixa(f"{BASE}/grids_151021_lhapdf/HERAPDF20_LO_EIG.tgz",
                    tmp / "lo.tgz")
        with tarfile.open(tgz) as t:
            t.extractall(tmp)
        extrai_pdfs(tmp / "HERAPDF20_LO_EIG" / "HERAPDF20_LO_EIG_0000.dat",
                    DESTINO / "herapdf20_lo_xf.dat")
    print(f"\npronto. arquivos em {DESTINO}")


if __name__ == "__main__":
    main()

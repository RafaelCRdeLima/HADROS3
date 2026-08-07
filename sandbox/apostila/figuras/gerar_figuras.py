#!/usr/bin/env python3
"""Gera todas as figuras da apostila de DIS.

Uso:  python3 gerar_figuras.py

Sai em PDF vetorial, com tipografia idêntica à do documento (backend pgf +
newpx), no mesmo diretório deste script. Os dados vêm de sandbox/data/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np

AQUI = Path(__file__).resolve().parent
DADOS = AQUI.parents[1] / "data"

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{newpxtext,newpxmath}",
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": "#3a3a38",
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.color": "#dcdcd6",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# paleta categórica em ordem fixa (nunca reciclada); ver sandbox/README.md
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CINZA, CINZA2 = "#3a3a38", "#7d7d76"

M_N = 0.9383          # massa do nucleon [GeV]
M_W = 80.377          # massa do W [GeV]
G_F = 1.1663788e-5    # constante de Fermi [GeV^-2]
GEV2_TO_CM2 = 3.893793e-28
M_BARYON_G = 1.67262192369e-24


def salvar(fig, nome):
    caminho = AQUI / nome
    fig.savefig(caminho, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  ->", caminho.name)


# ======================================================================
# dados
# ======================================================================
def carrega_hera_nc():
    d = np.loadtxt(DADOS / "hera" / "hera_nc_ep_920.dat")
    return dict(Q2=d[:, 0], x=d[:, 1], y=d[:, 2], sig=d[:, 3], err=d[:, 4])


def carrega_pdfs():
    d = np.loadtxt(DADOS / "hera" / "herapdf20_lo_xf.dat")
    saida = {}
    for Q2 in np.unique(d[:, 0]):
        m = d[:, 0] == Q2
        saida[float(Q2)] = dict(x=d[m, 1], uv=d[m, 2], dv=d[m, 3], ub=d[m, 4],
                                db=d[m, 5], s=d[m, 6], c=d[m, 7], b=d[m, 8], g=d[m, 9])
    return saida


def carrega_sigma_nu(modelo):
    d = np.loadtxt(DADOS / "sigma" / f"sigma_nuN_CC_{modelo}.dat")
    return d[:, 0], d[:, 2]


# ======================================================================
# fig 1 — plano cinemático (x, Q^2)
# ======================================================================
def fig_plano_cinematico():
    h = carrega_hera_nc()
    fig, ax = plt.subplots(figsize=(5.4, 3.9))

    # região de alvo fixo: sqrt(s) ~ 30 GeV  =>  Q^2 = x y s <= x s
    xs = np.logspace(-3, np.log10(0.8), 200)
    ax.fill_between(xs, 0.5, xs * 900.0, color=S[3], alpha=0.20, lw=0,
                    label=r"alvo fixo ($\sqrt{s}\sim 30$\,GeV)")

    ax.scatter(h["x"], h["Q2"], s=3.0, color=S[0], alpha=0.6, lw=0,
               label=r"HERA I+II, NC $e^+p$ (dados)")

    # linha cinemática do neutrino UHE: o propagador fixa Q^2 ~ m_W^2,
    # logo x ~ m_W^2/(2 m_N E). É a região que nenhum acelerador alcança.
    for E, rotulo in ((1e6, "10^{6}"), (1e9, "10^{9}"), (1e12, "10^{12}")):
        xt = M_W**2 / (2 * M_N * E)
        ax.plot([xt, xt], [0.12, 2e4], ls=":", lw=1.1, color=S[1])
        ax.text(xt * 1.5, 3e4, rf"$\nu$ UHE, $E_\nu\!=\!{rotulo}$", color=S[1],
                fontsize=6.4, ha="left", rotation=90, va="bottom")

    ax.axhline(M_W**2, color=S[2], lw=1.2, ls="--")
    ax.text(2e-9, M_W**2 * 1.3, r"$Q^2=m_W^2$", color=S[2], fontsize=7.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-9, 1.0)
    ax.set_ylim(0.1, 1e7)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$Q^2$\ \ [GeV$^2$]")
    ax.set_title(r"Onde cada experimento vive no plano $(x,Q^2)$")
    ax.legend(loc="upper right", handletextpad=0.5)
    salvar(fig, "fig_plano_cinematico.pdf")


# ======================================================================
# fig 2 — PDFs reais
# ======================================================================
def fig_pdfs():
    p = carrega_pdfs()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    for ax, Q2 in zip(axes, (10.0, 10000.0)):
        d = p[Q2]
        ax.plot(d["x"], d["uv"], color=S[0], label=r"$xu_v$")
        ax.plot(d["x"], d["dv"], color=S[1], label=r"$xd_v$")
        ax.plot(d["x"], 2 * (d["ub"] + d["db"] + d["s"] + d["c"] + d["b"]), color=S[2],
                label=r"$x\Sigma_{\rm mar}$")
        ax.plot(d["x"], d["g"] / 10.0, color=S[3], ls="--", label=r"$xg/10$")
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1.0)
        ax.set_ylim(0, 1.35)
        ax.set_xlabel(r"$x$")
        ax.set_title(rf"$Q^2 = {Q2:.0f}$\,GeV$^2$")
    axes[0].set_ylabel(r"$xf(x,Q^2)$")
    axes[0].legend(loc="upper left", ncol=2, columnspacing=1.0)
    salvar(fig, "fig_pdfs.pdf")


# ======================================================================
# fig 3 — violação de escala com dados reais
# ======================================================================
def fig_violacao_escala():
    h = carrega_hera_nc()
    alvos = [1.3e-4, 3.2e-4, 8.0e-4, 2.0e-3, 5.0e-3, 1.3e-2, 3.2e-2, 8.0e-2, 1.8e-1, 4.0e-1]
    fig, ax = plt.subplots(figsize=(4.9, 5.6))
    x_unicos = np.unique(h["x"])
    for i, alvo in enumerate(alvos):
        xb = x_unicos[np.argmin(abs(np.log(x_unicos) - np.log(alvo)))]
        m = (h["x"] == xb) & (h["y"] < 0.6)          # y baixo: sigma_red ~ F_2
        if m.sum() < 4:
            continue
        desl = 2.0 ** (len(alvos) - 1 - i)
        ordem = np.argsort(h["Q2"][m])
        Q2, sig, err = h["Q2"][m][ordem], h["sig"][m][ordem], h["err"][m][ordem]
        cor = S[i % len(S)]
        ax.errorbar(Q2, sig * desl, yerr=err * desl, fmt="o", ms=2.3, lw=0.7,
                    color=cor, mfc=cor, capsize=0)
        ax.plot(Q2, sig * desl, lw=0.7, color=cor, alpha=0.55)
        ax.text(Q2[-1] * 1.5, sig[-1] * desl, rf"$x={xb:.2g}$", fontsize=6.4,
                color=cor, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, 2e5)
    ax.set_xlabel(r"$Q^2$\ \ [GeV$^2$]")
    ax.set_ylabel(r"$\sigma_r\,(x,Q^2)\times 2^{\,i}$")
    ax.set_title("Violação de escala, medida\n" +
                 r"\small H1{\,}+{\,}ZEUS combinados, NC $e^+p$, $y<0.6$")
    salvar(fig, "fig_violacao_escala.pdf")


# ======================================================================
# fig 4 — dependência em y e o teste de spin
# ======================================================================
def fig_dependencia_y():
    y = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))

    ax = axes[0]
    ax.plot(y, np.ones_like(y), color=S[0], label=r"$\nu q$, $\bar\nu\bar q$:\ \ const.")
    ax.plot(y, (1 - y) ** 2, color=S[1], label=r"$\nu\bar q$, $\bar\nu q$:\ \ $(1-y)^2$")
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$\dfrac{\mathrm{d}\sigma}{\mathrm{d}y}$ (norm.)")
    ax.set_title("Parceiro de hélice: a assinatura V--A")
    ax.set_ylim(-0.03, 1.08)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(y, 1 + (1 - y) ** 2, color=S[0], label=r"$Y_+ = 1+(1-y)^2$")
    ax.plot(y, 1 - (1 - y) ** 2, color=S[1], label=r"$Y_- = 1-(1-y)^2$")
    ax.plot(y, y ** 2, color=S[2], ls="--", label=r"$y^2$ (peso de $F_L$)")
    ax.set_xlabel(r"$y$")
    ax.set_ylabel("peso")
    ax.set_title(r"Os pesos de $F_2$, $xF_3$ e $F_L$")
    ax.set_ylim(-0.05, 2.1)
    ax.legend(loc="upper center")
    salvar(fig, "fig_dependencia_y.pdf")


# ======================================================================
# fig 5 — propagador
# ======================================================================
def fig_propagador():
    Q2 = np.logspace(-1, 7, 400)
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ax.loglog(Q2, (M_W**2 / (M_W**2 + Q2)) ** 2, color=S[0],
              label=r"CC: $\left(\dfrac{m_W^2}{m_W^2+Q^2}\right)^{\!2}$")
    ax.loglog(Q2, (M_W**2 / Q2) ** 2, color=S[1], ls="--",
              label=r"sem massa: $\left(\dfrac{m_W^2}{Q^2}\right)^{\!2}$ (fóton)")
    ax.axvline(M_W**2, color=CINZA2, lw=0.9, ls=":")
    ax.text(M_W**2 * 1.35, 3e-7, r"$Q^2=m_W^2$", color=CINZA2, fontsize=7.2, rotation=90)
    ax.set_xlim(1e-1, 1e7)
    ax.set_ylim(1e-8, 1e5)
    ax.set_xlabel(r"$Q^2$\ \ [GeV$^2$]")
    ax.set_ylabel("fator de propagador")
    ax.set_title(r"A massa do $W$ corta a divergência em $Q^2\to 0$")
    ax.legend(loc="lower left")
    salvar(fig, "fig_propagador.pdf")


# ======================================================================
# fig 6 — sigma(E) do neutrino, dos aceleradores ao UHE
# ======================================================================
def fig_sigma_energia():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    # regime linear medido em aceleradores, valido ate ~350 GeV
    # (Formaggio & Zeller 2012): sigma/E = 0.677e-38 cm^2/GeV, alvo isoescalar
    E_lin = np.logspace(1, np.log10(350), 60)
    ax.loglog(E_lin, 0.677e-38 * E_lin, color=S[2], lw=2.6, alpha=0.9,
              label=r"medido: $\sigma/E = 0.677\times10^{-38}$\,cm$^2$/GeV")

    # GQRS so na faixa em que a parametrizacao foi declarada valida
    E_gq = np.logspace(4, 12, 300)
    ax.loglog(E_gq, 5.53e-36 * E_gq ** 0.363, color=S[3], ls="--",
              label=r"GQRS 1998: $5.53\times10^{-36}E^{0.363}$")
    for i, m in enumerate(("GBW", "IIM")):
        Eg, sg = carrega_sigma_nu(m)
        ax.loglog(Eg, sg, color=S[i], label=rf"{m} (tabela do HADROS3)")

    ax.axvline(M_W**2 / (2 * M_N), color=CINZA2, lw=0.9, ls=":")
    ax.text(M_W**2 / (2 * M_N) * 1.45, 1.5e-30, r"$s = m_W^2$", color=CINZA2,
            fontsize=7.2, rotation=90, va="top")
    ax.set_xlim(10, 1e14)
    ax.set_ylim(1e-38, 1e-29)
    ax.set_xlabel(r"$E_\nu$\ \ [GeV] (alvo em repouso)")
    ax.set_ylabel(r"$\sigma^{\rm CC}_{\nu N}$\ \ [cm$^2$]")
    ax.set_title("A quebra do crescimento linear")
    # canto inferior direito: é a única região do gráfico que fica vazia
    ax.legend(loc="lower right")
    salvar(fig, "fig_sigma_energia.pdf")


# ======================================================================
# fig 7 — x típico e resolução
# ======================================================================
def fig_x_tipico():
    E = np.logspace(1, 14, 400)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))

    ax = axes[0]
    ax.loglog(E, M_W**2 / (2 * M_N * E), color=S[0])
    ax.axhline(1.0, color=CINZA2, lw=0.9, ls=":")
    ax.axhspan(1e-6, 1e-2, color=S[0], alpha=0.10, lw=0)
    ax.text(3e10, 2e-5, "região de $x$ pequeno:\nsem dados diretos",
            fontsize=6.8, color=CINZA, ha="center")
    ax.set_xlim(10, 1e14)
    ax.set_ylim(1e-9, 30)
    ax.set_xlabel(r"$E_\nu$\ \ [GeV]")
    ax.set_ylabel(r"$x_{\rm tip} \simeq m_W^2/2m_N E_\nu$")
    ax.set_title(r"O $x$ que o neutrino sonda")

    ax = axes[1]
    Q2 = np.logspace(-2, 6, 300)
    hbar_c_fm = 0.19733                          # GeV fm
    ax.loglog(Q2, hbar_c_fm / np.sqrt(Q2), color=S[0])
    for val, rot in ((0.84, "raio do próton"), (1e-3, "escala do quark?")):
        ax.axhline(val, color=CINZA2, lw=0.8, ls=":")
        ax.text(2e-2, val * 1.25, rot, fontsize=6.8, color=CINZA)
    ax.set_xlim(1e-2, 1e6)
    ax.set_xlabel(r"$Q^2$\ \ [GeV$^2$]")
    ax.set_ylabel(r"$\lambda \sim \hbar c/\sqrt{Q^2}$\ \ [fm]")
    ax.set_title("A régua do experimento")
    salvar(fig, "fig_x_tipico.pdf")


# ======================================================================
# fig 8 — modelo de pártons vs dados: F_2 a partir das PDFs reais
# ======================================================================
def fig_f2_modelo_partons():
    p = carrega_pdfs()
    h = carrega_hera_nc()
    Q2 = 100.0
    d = p[Q2]
    # F_2^{em} = sum_q e_q^2 x(q + qbar) a LO. Somamos u, d, s, c, b -- o charm
    # NAO e desprezivel em x pequeno, e omiti-lo derruba a curva visivelmente.
    F2_leves = (4 / 9) * (d["uv"] + 2 * d["ub"]) + (1 / 9) * (d["dv"] + 2 * d["db"]) \
               + (1 / 9) * (2 * d["s"])
    F2 = F2_leves + (4 / 9) * (2 * d["c"]) + (1 / 9) * (2 * d["b"])

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(d["x"], F2, color=S[0], lw=1.6,
            label=r"$F_2 = \sum_q e_q^2\,x(q+\bar q)$, HERAPDF2.0 LO")
    ax.plot(d["x"], F2_leves, color=S[2], lw=1.1, ls="--",
            label=r"idem, sem $c$ e $b$")

    m = (abs(np.log(h["Q2"] / Q2)) < 0.18) & (h["y"] < 0.6)
    ordem = np.argsort(h["x"][m])
    ax.errorbar(h["x"][m][ordem], h["sig"][m][ordem], yerr=h["err"][m][ordem],
                fmt="o", ms=3.2, lw=0.8, color=S[1], capsize=0,
                label=r"$\sigma_r$ medido, $Q^2\simeq 100$\,GeV$^2$")

    ax.set_xscale("log")
    ax.set_xlim(1e-4, 0.7)
    ax.set_ylim(0, 1.85)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$F_2$,\ \ $\sigma_r$")
    ax.set_title("O modelo de pártons contra o experimento")
    ax.legend(loc="lower left")
    salvar(fig, "fig_f2_modelo_partons.pdf")


# ======================================================================
# fig 9 — a integral de profundidade óptica (elo com o notebook 01)
# ======================================================================
def fig_profundidade_optica():
    Eg, sg = carrega_sigma_nu("GBW")
    E = np.logspace(3, 14, 300)
    sigma = np.exp(np.interp(np.log(E), np.log(Eg), np.log(sg)))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    cenarios = [("Terra (diâmetro)", 5.51 * 1.2742e9, S[0]),
                ("Sol (diâmetro)", 1.41 * 1.392e11, S[1]),
                ("toro do HADROS3 (equador)", 3.03e16, S[2])]
    for nome, X, cor in cenarios:
        ax.loglog(E, X / M_BARYON_G * sigma, color=cor, label=rf"{nome}")
    ax.axhline(1.0, color=CINZA2, lw=1.0, ls=":")
    ax.text(2e3, 1.6, r"$\tau=1$", color=CINZA2, fontsize=7.4)
    ax.set_xlim(1e3, 1e14)
    ax.set_ylim(1e-4, 1e12)
    ax.set_xlabel(r"$E_\nu$\ \ [GeV]")
    ax.set_ylabel(r"$\tau = \sigma X/m_b$")
    ax.set_title(r"Da seção de choque à opacidade: $\tau$ para três colunas")
    ax.legend(loc="lower right")
    salvar(fig, "fig_profundidade_optica.pdf")


if __name__ == "__main__":
    print("gerando figuras em", AQUI)
    fig_plano_cinematico()
    fig_pdfs()
    fig_violacao_escala()
    fig_dependencia_y()
    fig_propagador()
    fig_sigma_energia()
    fig_x_tipico()
    fig_f2_modelo_partons()
    fig_profundidade_optica()
    print("pronto.")

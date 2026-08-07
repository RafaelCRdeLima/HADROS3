#!/usr/bin/env python3
"""Regenera dis.bib buscando o BibTeX exato de cada referência no INSPIRE-HEP.

Uso:  python3 buscar_referencias.py

Nenhuma entrada da bibliografia é digitada à mão: cada uma vem da API do
INSPIRE (https://inspirehep.net/api/literature?q=...&format=bibtex), o que
elimina a chance de inventar volume, página ou ano. Se uma consulta falhar,
ela é reportada e o script segue --- confira o resultado antes de commitar.
"""

import subprocess
import time
import urllib.parse
from pathlib import Path

CONSULTAS = [
    # ---- clássicas ----
    ("bjorken1969", 'a Bjorken and t "Asymptotic Sum Rules at Infinite Momentum"'),
    ("bjorken_paschos1969", 't "Inelastic Electron - Proton and gamma - Proton Scattering, and the Structure of the Nucleon"'),
    ("feynman1969", 't "Very High-Energy Collisions of Hadrons"'),
    ("callan_gross1969", 't "High-energy electroproduction and the constitution of the electric current"'),
    ("bloom1969", 't "High-Energy Inelastic e p Scattering at 6-Degrees and 10-Degrees"'),
    ("breidenbach1969", 't "Observed Behavior of Highly Inelastic electron-Proton Scattering"'),
    ("gross_wilczek1973", 't "Ultraviolet Behavior of Nonabelian Gauge Theories"'),
    ("politzer1973", 't "Reliable Perturbative Results for Strong Interactions?"'),
    ("altarelli_parisi1977", 't "Asymptotic Freedom in Parton Language"'),
    ("gribov_lipatov1972", 'a Gribov and a Lipatov and t "Deep inelastic e p scattering in perturbation theory"'),
    ("dokshitzer1977", 'a Dokshitzer and t "Calculation of the Structure Functions for Deep Inelastic Scattering and e+ e- Annihilation by Perturbation Theory in Quantum Chromodynamics"'),
    ("friedman_kendall1972", 'a Friedman and a Kendall and t "Deep inelastic electron scattering"'),
    # ---- modernas ----
    ("hera_combined2015", "arxiv:1506.06042"),
    ("gbw1998", "arxiv:hep-ph/9807513"),
    ("gbw1999", "arxiv:hep-ph/9903358"),
    ("iim2004", "arxiv:hep-ph/0310338"),
    ("gqrs1998", "arxiv:hep-ph/9807264"),
    ("cooper_sarkar2011", "arxiv:1106.3723"),
    ("connolly2011", "arxiv:1102.0691"),
    ("formaggio_zeller2012", "arxiv:1305.7513"),
    ("bertone2018", "arxiv:1808.02034"),
    ("icecube2017xsec", "arxiv:1711.08119"),
    ("eic_yellow2021", "arxiv:2103.05419"),
    ("eic_whitepaper2012", "arxiv:1212.1701"),
    ("balitsky1996", 'a Balitsky and t "Operator expansion for high-energy scattering"'),
    ("kovchegov1999", 'a Kovchegov and t "Small x F(2) structure function of a nucleus including multiple pomeron exchanges"'),
    ("bfkl1978", 'a Balitsky and a Lipatov and t "The Pomeranchuk Singularity in Quantum Chromodynamics"'),
    # ---- livros e textos didáticos ----
    ("halzen_martin", 'a Halzen and t "Quarks and Leptons: An Introductory Course in Modern Particle Physics"'),
    ("peskin_schroeder", 'a Peskin and t "An Introduction to quantum field theory"'),
    ("esw_qcd", 'a Ellis and a Stirling and a Webber and t "QCD and collider physics"'),
    ("devenish_cooper_sarkar", 'a Devenish and t "Deep inelastic scattering"'),
    ("close_quarks_partons", 'a Close and t "An Introduction to Quarks and Partons"'),
    ("thomson_modern", 'a Thomson and t "Modern particle physics"'),
    ("schwartz_qft", 'a Schwartz and t "Quantum Field Theory and the Standard Model"'),
    ("collins_foundations", 'a Collins and t "Foundations of perturbative QCD"'),
    ("roberts_proton", 'a Roberts and t "The Structure of the proton: Deep inelastic scattering"'),
    ("griffiths_elementary", 'a Griffiths and t "Introduction to elementary particles"'),
    ("pdg", 'cn "Particle Data Group" and t "Review of Particle Physics"'),
]

OUT = Path(__file__).resolve().parent / "dis.bib"
partes = ["""% Referências da apostila de DIS do HADROS Sandbox.
% NÃO EDITE À MÃO: este arquivo é gerado por buscar_referencias.py, que puxa
% cada entrada da API BibTeX do INSPIRE-HEP.
"""]
for rotulo, consulta in CONSULTAS:
    # só o PDG quer o mais recente; nas demais, o INSPIRE já ordena por
    # relevância e pedir "mostrecent" traz o artigo errado.
    ordem = "&sort=mostrecent" if rotulo == "pdg" else ""
    url = ("https://inspirehep.net/api/literature?q="
           + urllib.parse.quote(consulta) + f"&size=1{ordem}&format=bibtex")
    r = subprocess.run(["curl", "-sS", "--max-time", "40", url],
                       capture_output=True, text=True)
    txt = r.stdout.strip()
    ok = txt.startswith("@")
    print(f"{'OK ' if ok else 'FALHOU'} {rotulo:<26} {txt.splitlines()[0][:70] if txt else '(vazio)'}")
    if ok:
        partes.append(f"% ==== consulta: {rotulo} ====\n{txt}\n")
    time.sleep(0.4)

OUT.write_text("\n".join(partes), encoding="utf-8")
print(f"\n{len(partes)-1}/{len(CONSULTAS)} entradas -> {OUT}")

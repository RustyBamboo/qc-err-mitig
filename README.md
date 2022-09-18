<div align = center>
<img src="https://raw.githubusercontent.com/rustybamboo/qc-err-mitig/main/logo.png" alt="banner">
<br>

<br>
<h3><a href="https://rustybamboo.github.io/qc-err-mitig">https://rustybamboo.github.io/qc-err-mitig</a></h3>

<br>

</div>

---

## Tutorial on Quantum Noise Characterization and Mitigation

To view the rendered tutorial, please click the link above.

### Running Locally

```
git clone --depth 1 https://github.com/RustyBamboo/qc-err-mitig
cd qc-err-mitig
```

Create a conda environment 
```
conda env create -f environment.yml
```

Ensure jupyter server has [jupytext](https://jupytext.readthedocs.io/en/latest/install.html) extension loaded:

```
jupyter serverextension enable jupytext
```

Then run the jupyter notebook:
```
jupyter notebook
```

> If you don't want to use conda, then you can run everything directly via your own installation of jupyter notebook and jupytext extension.

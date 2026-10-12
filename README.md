# GOPREAUX
***Gaussian process Optimized Photometric Regression of Extragalactic Archival Ultraviolet-infrared eXplosions***

A package for multi-dimensional Gaussian Process Regression of extragalactic astronomical transient light curves, enabling a full characterization of their spectral energy distribution evolution with time.

## Background
Time-domain astronomy is entering a golden age of discovery, powered by wide-field surveys such as Rubin Observatory's Legacy Survey of Space and Time. The wealth of data that will be produced over the coming years offers time-domain astronomers the opportunity to conduct population-level studies of different types of explosive extragalactic transients, many for the first time. However, our knowledge of the physics underpinning these explosions is lagging, and parsing the deluge of data in real time to identify interesting transients requires detailed knowledge of their time evolution.

`GOPREAUX` addresses these problems by producing multidimensional template light curve and spectral surfaces of different classes of transients for the first time. `GOPREAUX` models aggregated archival data, spanning the ultraviolet to the infrared, using Gaussian Process Regression across both phase and wavelength simultaneously. The models produce "template" time-evolving spectral energy distribution surfaces that can be used for a variety of use cases--to perform physical parameter inference, generate machine learning training sets of different transients at arbitrary phase and redshift, or identify rare and unusual transients in real time, to name a few use cases. 

## Installation and Setup
Dependencies are managed using `poetry`. The recommended installation is to create a new python environment for this repo and install `poetry` within that environment.

```
> conda create -n gopreaux python=3.11
> conda activate gopreaux
> pip install poetry
```
`poetry` can then be used to install `GOPREAUX` and its dependencies. First, clone this repository:

```
> git clone https://github.com/crpellegrino/gopreaux.git
```
Then from the root of the repository, run the installation:
```
> cd gopreaux
> poetry install
```

Finally, to make use of Milky Way extinction correction, the dust map files of Schlafly and Finkbeiner (2011) must be fetched and saved locally. To do so, open a Python shell and run:

```py
import dustmaps.sfd
dustmaps.sfd.fetch()
```

That's it! `gopreaux` should now be ready to use.

**Note**: Recent changes to the Harvard Dataverse API have broken the `dustmaps` `fetch()` methods. To work around this, manually download the SFD FITS files from the Harvard Dataverse and place them inside the `data/sfd/` subdirectory within your `dustmaps` installation (i.e. inside your conda environment).

## Data Structure

```mermaid

flowchart TD
    A{Data};
    subgraph Types;
        A---->B[FBOT];
        A---->C[SESNe];
        A---->D[SLSN-I];
        A---->E[SLSN-II];
        A---->F[SNII];
        A---->G[SNIIn];
        A---->b[SNIa];
        A---->a[Other];
    end;
    subgraph Subtypes;
        B---->H(SNIbn);
        B---->I(SNIcn);
        C---->J(SNIIb);
		C---->V(SNIb);
		C---->W(SNIc);
        D---->K(SLSN-II);
        E---->L(SLSN-I);
        F---->M(SNIIP);
        F---->N(SNII-pec);
        F---->O(SNII);
        G---->P(SNIIn);
        b---->Q(SNIa);
        b---->R(SNIa-pec);
		a---->X(TDE);
		a---->Y(Unclassified/Other);
    end;
    S{Data}-->T[Type]-->U(Subtype);

    style H fill:#e0f8f8,stroke:#2E86C1
    style I fill:#e0f8f8,stroke:#2E86C1
    style J fill:#e0f8f8,stroke:#2E86C1
	style V fill:#e0f8f8,stroke:#2E86C1
	style W fill:#e0f8f8,stroke:#2E86C1
    style K fill:#e0f8f8,stroke:#2E86C1
    style L fill:#e0f8f8,stroke:#2E86C1
    style M fill:#e0f8f8,stroke:#2E86C1
    style N fill:#e0f8f8,stroke:#2E86C1
    style O fill:#e0f8f8,stroke:#2E86C1
    style P fill:#e0f8f8,stroke:#2E86C1
    style Q fill:#e0f8f8,stroke:#2E86C1
    style R fill:#e0f8f8,stroke:#2E86C1
	style X fill:#e0f8f8,stroke:#2E86C1
	style Y fill:#e0f8f8,stroke:#2E86C1
    style U fill:#e0f8f8,stroke:#2E86C1
    style A fill:#daf7a6,stroke:#1E8449
    style S fill:#daf7a6,stroke:#1E8449
```
The above mermaid chart details the out-of-the-box organization of transient data into types and subtypes. The structure was chosen to group transients based on similar light curve and SED behavior, rather than physical characteristics such as powering mechanisms or progenitor channels.

## Code Example
```py
from caat import SN, SNModel

# Load the GP model
model = SNModel(
    surface="SESNe_SNIIb_GP_model.fits"
)

# Predict a light curve from -20 to 45 days at 5000 Angstroms
model.predict_lightcurve(-20.0, 45.0, 5000, show=False)

# Compare the prediction to real photometry from a supernova
gkg = SN(name="SN2016gkg")
model.compare_lightcurve_with_photometry(gkg, filt="V")
```

More examples can be found as Jupyter notebooks within the `examples/` directory.
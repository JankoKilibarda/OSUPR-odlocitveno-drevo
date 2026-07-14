"""
OSUPR projekt: preprosto odločitveno drevo za binarno klasifikacijo

Potrebne knjižnice:
    numpy
    pandas
    scikit-learn

Datoteka titanic.csv mora biti v isti mapi kot ta program.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class Vozlisce:
    """Eno vozlišče odločitvenega drevesa."""

    def __init__(
        self,
        atribut=None,
        prag=None,
        levo=None,
        desno=None,
        razred=None,
    ):
        self.atribut = atribut
        self.prag = prag
        self.levo = levo
        self.desno = desno
        self.razred = razred

    def je_list(self):
        return self.razred is not None


class MojeOdlocitvenoDrevo:
    """
    Preprosta lastna implementacija odločitvenega drevesa.

    Podpira:
    - Gini indeks,
    - entropijo,
    - maksimalno globino,
    - minimalno število primerov v listu.
    """

    def __init__(self, kriterij="gini", maksimalna_globina=4, najmanj_v_listu=5):
        if kriterij not in ("gini", "entropy"):
            raise ValueError("Kriterij mora biti 'gini' ali 'entropy'.")

        self.kriterij = kriterij
        self.maksimalna_globina = maksimalna_globina
        self.najmanj_v_listu = najmanj_v_listu
        self.koren = None

    def fit(self, X, y):
        """Zgradi drevo iz učnih podatkov."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.koren = self._zgradi_drevo(X, y, globina=0)
        return self

    def predict(self, X):
        """Napove razred za vsak primer."""
        X = np.asarray(X, dtype=float)
        return np.array([self._napovej_eno(vrstica, self.koren) for vrstica in X])

    def _necistost_iz_stevila(self, negativni, pozitivni):
        """Izračuna Gini ali entropijo iz števila primerov obeh razredov."""
        skupaj = negativni + pozitivni

        if skupaj == 0:
            return 0.0

        p0 = negativni / skupaj
        p1 = pozitivni / skupaj

        if self.kriterij == "gini":
            return 1.0 - p0**2 - p1**2

        entropija = 0.0
        for p in (p0, p1):
            if p > 0:
                entropija -= p * np.log2(p)
        return entropija

    def _vecinski_razred(self, y):
        """Vrne najpogostejši razred."""
        return int(np.mean(y) >= 0.5)

    def _najboljsi_razcep(self, X, y):
        """Poišče najboljši atribut in prag razcepa."""
        stevilo_primerov, stevilo_atributov = X.shape

        pozitivni_skupaj = int(np.sum(y))
        negativni_skupaj = stevilo_primerov - pozitivni_skupaj
        zacetna_necistost = self._necistost_iz_stevila(
            negativni_skupaj,
            pozitivni_skupaj,
        )

        najboljsi_dobicek = 0.0
        najboljsi_atribut = None
        najboljsi_prag = None

        for atribut in range(stevilo_atributov):
            vrstni_red = np.argsort(X[:, atribut])
            x_urejen = X[vrstni_red, atribut]
            y_urejen = y[vrstni_red]

            pozitivni_levo = 0

            for i in range(stevilo_primerov - 1):
                pozitivni_levo += int(y_urejen[i])

                # Če sta sosednji vrednosti enaki, med njima ni smiselnega praga.
                if x_urejen[i] == x_urejen[i + 1]:
                    continue

                levo_n = i + 1
                desno_n = stevilo_primerov - levo_n

                if (
                    levo_n < self.najmanj_v_listu
                    or desno_n < self.najmanj_v_listu
                ):
                    continue

                pozitivni_desno = pozitivni_skupaj - pozitivni_levo
                negativni_levo = levo_n - pozitivni_levo
                negativni_desno = desno_n - pozitivni_desno

                necistost_levo = self._necistost_iz_stevila(
                    negativni_levo,
                    pozitivni_levo,
                )
                necistost_desno = self._necistost_iz_stevila(
                    negativni_desno,
                    pozitivni_desno,
                )

                tehtana_necistost = (
                    levo_n / stevilo_primerov * necistost_levo
                    + desno_n / stevilo_primerov * necistost_desno
                )

                dobicek = zacetna_necistost - tehtana_necistost

                if dobicek > najboljsi_dobicek:
                    najboljsi_dobicek = dobicek
                    najboljsi_atribut = atribut
                    najboljsi_prag = (x_urejen[i] + x_urejen[i + 1]) / 2

        return najboljsi_atribut, najboljsi_prag

    def _zgradi_drevo(self, X, y, globina):
        """Rekurzivno zgradi drevo."""
        samo_en_razred = len(np.unique(y)) == 1
        dosezena_globina = globina >= self.maksimalna_globina
        premalo_primerov = len(y) < 2 * self.najmanj_v_listu

        if samo_en_razred or dosezena_globina or premalo_primerov:
            return Vozlisce(razred=self._vecinski_razred(y))

        atribut, prag = self._najboljsi_razcep(X, y)

        if atribut is None:
            return Vozlisce(razred=self._vecinski_razred(y))

        levo_maska = X[:, atribut] <= prag
        desno_maska = ~levo_maska

        levo = self._zgradi_drevo(
            X[levo_maska],
            y[levo_maska],
            globina + 1,
        )
        desno = self._zgradi_drevo(
            X[desno_maska],
            y[desno_maska],
            globina + 1,
        )

        return Vozlisce(
            atribut=atribut,
            prag=prag,
            levo=levo,
            desno=desno,
        )

    def _napovej_eno(self, vrstica, vozlisce):
        """Napove razred za en primer."""
        if vozlisce.je_list():
            return vozlisce.razred

        if vrstica[vozlisce.atribut] <= vozlisce.prag:
            return self._napovej_eno(vrstica, vozlisce.levo)

        return self._napovej_eno(vrstica, vozlisce.desno)


def pripravi_titanic():
    """Prebere in preprosto pripravi podatke Titanic."""
    pot = Path(__file__).with_name("titanic.csv")

    if not pot.exists():
        raise FileNotFoundError(
            "Datoteka titanic.csv mora biti v isti mapi kot program."
        )

    podatki = pd.read_csv(pot)

    # Uporabimo samo pregledne in uporabne atribute.
    atributi = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

    podatki["Sex"] = podatki["Sex"].map({"male": 0, "female": 1})
    podatki["Age"] = podatki["Age"].fillna(podatki["Age"].median())
    podatki["Fare"] = podatki["Fare"].fillna(podatki["Fare"].median())

    X = podatki[atributi].to_numpy(dtype=float)
    y = podatki["Survived"].to_numpy(dtype=int)

    return X, y


def pripravi_rak():
    """Naloži Breast Cancer Wisconsin podatke."""
    podatki = load_breast_cancer()
    return podatki.data, podatki.target


def preizkusi_podatke(ime, X, y):
    """Nauči naše in sklearn drevo ter izpiše rezultate."""
    X_ucni, X_testni, y_ucni, y_testni = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\n" + "=" * 60)
    print(ime)
    print("=" * 60)

    for kriterij in ("gini", "entropy"):
        moje_drevo = MojeOdlocitvenoDrevo(
            kriterij=kriterij,
            maksimalna_globina=4,
            najmanj_v_listu=5,
        )
        moje_drevo.fit(X_ucni, y_ucni)
        moje_napovedi = moje_drevo.predict(X_testni)
        moja_natancnost = accuracy_score(y_testni, moje_napovedi)

        sklearn_drevo = DecisionTreeClassifier(
            criterion=kriterij,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
        )
        sklearn_drevo.fit(X_ucni, y_ucni)
        sklearn_napovedi = sklearn_drevo.predict(X_testni)
        sklearn_natancnost = accuracy_score(y_testni, sklearn_napovedi)

        print(f"\nKriterij: {kriterij}")
        print(f"Moje drevo:    {moja_natancnost:.4f}")
        print(f"Sklearn drevo: {sklearn_natancnost:.4f}")


def main():
    print("OSUPR PROJEKT: ODLOČITVENO DREVO")

    X_rak, y_rak = pripravi_rak()
    preizkusi_podatke("Breast Cancer Wisconsin", X_rak, y_rak)

    X_titanic, y_titanic = pripravi_titanic()
    preizkusi_podatke("Titanic", X_titanic, y_titanic)

    print("\nProgram je uspešno končan.")


if __name__ == "__main__":
    main()

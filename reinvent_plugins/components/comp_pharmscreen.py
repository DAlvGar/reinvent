from __future__ import annotations
import os
import subprocess
import tempfile
from typing import List, Optional
import shutil
import numpy as np
from pydantic.dataclasses import dataclass
from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles
from rdkit import Chem
import time
import json
import logging

# os.environ["PSM3_DISABLE"] = "1"
# os.environ["FI_PROVIDER"] = "tcp"
#export _JAVA_OPTIONS="-XX:ParallelGCThreads=8 -XX:CICompilerCount=8"

logger = logging.getLogger("reinvent")

#########################################
# Parameters
#########################################

@add_tag("__parameters")
@dataclass
class Parameters:
    reference_file: List[str]
    pharmscreen_exec: List[str]
    license_file: List[str]
    reference_prepared_dir: List[str]
    threads: Optional[List[int]] = None
    logging_dir: Optional[List[str]] = None
    chemaxon_protonate_tautomerize: Optional[List[bool]] = False
    chemaxon_command: Optional[List[str]] = None
    protonation_pH: Optional[List[float]] = 7.4
    keep_temp: Optional[List[bool]] = False
    generate_conformers_chemaxon: Optional[List[bool]] = False


#########################################
# Main scoring component
#########################################

@add_tag("__component")
class PharmScreenSimilarity:

    def __init__(self, params: Parameters):
        logger.info("Initializing PharmScreenSimilarity component.")

        self.reference_file = params.reference_file[0]
        self.pharmscreen_exec = params.pharmscreen_exec[0]
        self.license_file = params.license_file[0]
        self.prepared_ref_dir = params.reference_prepared_dir[0]
        self.threads = params.threads[0] if params.threads else None
        self.logging_dir = params.logging_dir[0] if params.logging_dir else None
        self.keep_temp = params.keep_temp[0]

        self.chemaxon_mode = params.chemaxon_protonate_tautomerize[0]
        self.chemaxon_command = params.chemaxon_command[0] if params.chemaxon_command else None
        self.protonation_pH = params.protonation_pH[0]

        self.generate_conformers_chemaxon = params.generate_conformers_chemaxon[0]

        self.smiles_type = "rdkit_smiles"

        self.prepared_reference = self._prepare_reference()
        self.run_counter = 0

    #########################################
    # Public call
    #########################################

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> ComponentResults:
        self.run_counter += 1
        logger.info(f"Starting PharmScreen run {self.run_counter} with {len(smilies)} molecules.")

        timings = {}
        t0 = time.perf_counter()

        temp_dir = self._get_temp_dir()
        run_log_dir = self._get_run_log_dir()

        prepared_mols, processed_smiles = self._prepare_input_molecules(smilies, temp_dir, timings)
        self._log_input_smiles(run_log_dir, smilies, processed_smiles)

        lib_sdf = os.path.join(temp_dir, "library.sdf")
        self._write_library_sdf(prepared_mols, lib_sdf)
        logger.info("Library SDF written.")

        try:
            lib_prepared = self._prepare_library_with_pharmscreen(lib_sdf, temp_dir, timings)
            ranking_file = self._run_virtual_screening(lib_prepared, temp_dir, timings)
            scores = self._extract_scores(ranking_file, len(smilies))
        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout occurred during execution: {e}")
            scores = np.zeros(len(smilies))  # Return zeros for all input molecules

        timings["total"] = time.perf_counter() - t0
        logger.info(f"PharmScreen run {self.run_counter} completed. Total time: {timings['total']:.2f} sec.")

        self._save_timings(run_log_dir, timings)

        return ComponentResults([scores])

    #########################################
    # Step 1: Reference preparation
    #########################################

    def _prepare_reference(self) -> str:
        prepared_sdf = os.path.join(self.prepared_ref_dir, "prepared_reference.sdf")
        if os.path.exists(prepared_sdf):
            logger.debug("Using cached prepared reference.")
            return prepared_sdf

        os.makedirs(self.prepared_ref_dir, exist_ok=True)
        ref_outdir = os.path.join(self.prepared_ref_dir, "ref_prep_out")
        os.makedirs(ref_outdir, exist_ok=True)

        input_base = os.path.splitext(os.path.basename(self.reference_file))[0]
        ref_output = os.path.join(ref_outdir, f"{input_base}_GChg_ATLogP.sdf")

        if not os.path.exists(ref_output):
            logger.info("Preparing reference molecule(s) with PharmScreen.")
            cmd = [
                self.pharmscreen_exec,
                "-i", self.reference_file,
                "-x", "ref_preparation",
                "--ligprep",
                "-y", "gasteiger",
                "--logp", "at",
                "--key", self.license_file,
                "-p", ref_outdir
            ]
            logger.debug(f"Running command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, timeout=300)  # Adjusted timeout to 300 seconds (5 minutes)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed: {e}")
                raise
            except subprocess.TimeoutExpired as e:
                logger.error(f"Command timed out: {e}")
                raise

        shutil.copy(ref_output, prepared_sdf)
        return prepared_sdf

    #########################################
    # Helper functions: __call__ steps
    #########################################

    def _get_temp_dir(self) -> str:
        if self.keep_temp:
            temp_dir = tempfile.mkdtemp()
            self._temp_dir_obj = None
        else:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = self._temp_dir_obj.name
        return temp_dir


    def _get_run_log_dir(self) -> Optional[str]:
        if self.logging_dir:
            os.makedirs(self.logging_dir, exist_ok=True)
            run_log_dir = os.path.join(self.logging_dir, f"run_{self.run_counter:04d}")
            os.makedirs(run_log_dir, exist_ok=True)
            return run_log_dir
        return None

    def _prepare_input_molecules(self, smilies: List[str], temp_dir: str, timings: dict):
        t_start = time.perf_counter()
        if self.chemaxon_mode:
            logger.info("Starting ChemAxon protonation, tautomerization, and optionally conformer generation.")
            mols = self._chemaxon_protonate_tautomerize(smilies, temp_dir)
            smiles = []
            for mol_list in mols:
                if mol_list:
                    smiles.append(Chem.MolToSmiles(mol_list[0]))
                else:
                    smiles.append("")
        else:
            mols = []
            smiles = []
            for smi in smilies:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    mol = self._protonate_and_tautomerize(mol)
                    mols.append([mol])
                    smiles.append(Chem.MolToSmiles(mol))
                else:
                    mols.append([])
                    smiles.append("")
        timings["protonate_tautomerize"] = time.perf_counter() - t_start
        logger.info(f"Protonation/tautomerization completed in {timings['protonate_tautomerize']:.2f} sec.")
        return mols, smiles

    def _log_input_smiles(self, run_log_dir: Optional[str], original: List[str], processed: List[str]):
        if not run_log_dir:
            return
        with open(os.path.join(run_log_dir, "input_original.smi"), "w") as f:
            for smi in original:
                f.write(smi + "\n")
        with open(os.path.join(run_log_dir, "input_processed.smi"), "w") as f:
            for smi in processed:
                f.write(smi + "\n")

    def _write_library_sdf(self, mols: List[List[Chem.Mol]], sdf_path: str):
        writer = Chem.SDWriter(sdf_path)
        for idx, mol_list in enumerate(mols):
            for mol in mol_list:
                mol.SetProp("_Name", f"mol{idx:05d}")
                writer.write(mol)
        writer.close()

    def _prepare_library_with_pharmscreen(self, lib_sdf: str, temp_dir: str, timings: dict) -> str:
        t_start = time.perf_counter()
        libprep_out = os.path.join(temp_dir, "lib_preparation")
        os.makedirs(libprep_out, exist_ok=True)

        input_base = os.path.splitext(os.path.basename(lib_sdf))[0]

        prep_cmd = [
            self.pharmscreen_exec,
            "-i", lib_sdf,
            "-x", "lib_preparation",
            "--ligprep",
            "-y", "gasteiger",
            "--logp", "at",
            "--key", self.license_file,
            "-p", libprep_out
        ]

        if not self.generate_conformers_chemaxon:
            prep_cmd.append("--genconf")
            prep_cmd.append("--genstereo")
            suffix = "_Conf_GChg_ATLogP.sdf"
        else:
            suffix = "_GChg_ATLogP.sdf"

        if self.threads:
            prep_cmd.extend(["--threads", str(self.threads)])

        logger.info("Running PharmScreen library preparation.")
        logger.debug(f"PharmScreen command: {' '.join(prep_cmd)}")

        log_file = os.path.join(libprep_out, "pharmscreen_libprep.log")
        try:
            with open(log_file, "w") as logfile:
                subprocess.run(prep_cmd, stdout=logfile, stderr=logfile, check=True, timeout=180)  # Adjusted timeout to 180 seconds (3 minutes)
        except subprocess.CalledProcessError as e:
            logger.error(f"PharmScreen command failed: {e}")
            raise
        except subprocess.TimeoutExpired as e:
            logger.error(f"PharmScreen command timed out: {e}")
            raise

        timings["library_preparation"] = time.perf_counter() - t_start
        logger.info(f"Library preparation completed in {timings['library_preparation']:.2f} sec.")

        return os.path.join(libprep_out, input_base + suffix)

    def _run_virtual_screening(self, lib_prepared: str, temp_dir: str, timings: dict) -> str:
        t_start = time.perf_counter()
        vs_out = os.path.join(temp_dir, "vs")
        os.makedirs(vs_out, exist_ok=True)

        vs_cmd = [
            self.pharmscreen_exec,
            "-i", lib_prepared,
            "-k", self.prepared_reference,
            "-x", "vs",
            "--logp", "userdefined",
            "--key", self.license_file,
            "-p", vs_out
        ]

        if self.threads:
            vs_cmd.extend(["--threads", str(self.threads)])

        logger.info("Running PharmScreen virtual screening.")
        logger.debug(f"PharmScreen VS command: {' '.join(vs_cmd)}")

        log_file = os.path.join(vs_out, "pharmscreen_vs.log")
        try:
            with open(log_file, "w") as logfile:
                subprocess.run(vs_cmd, stdout=logfile, stderr=logfile, check=True, timeout=180)  # Adjusted timeout to 180 seconds (3 minutes)
        except subprocess.CalledProcessError as e:
            logger.error(f"PharmScreen command failed: {e}")
            raise
        except subprocess.TimeoutExpired as e:
            logger.error(f"PharmScreen command timed out: {e}")
            raise

        timings["virtual_screening"] = time.perf_counter() - t_start
        logger.info(f"Virtual screening completed in {timings['virtual_screening']:.2f} sec.")

        return os.path.join(vs_out, "ranking.csv")

    def _extract_scores(self, ranking_file: str, n_mols: int) -> np.ndarray:
        import pandas as pd
        df = pd.read_csv(ranking_file, sep=";")
        scores = np.full(n_mols, np.nan)
        for _, row in df.iterrows():
            name = str(row["Molecule_Name"])
            if name.startswith("mol"):
                mol_idx = int(name[3:8])
                if mol_idx < n_mols:
                    scores[mol_idx] = row["SimilarityScore"]
        return scores

    def _save_timings(self, run_log_dir: Optional[str], timings: dict):
        if run_log_dir:
            with open(os.path.join(run_log_dir, "timings.json"), "w") as f:
                json.dump(timings, f, indent=4)

    #########################################
    # ChemAxon protonate, tautomerize (+ conformers)
    #########################################

    def _protonate_and_tautomerize(self, mol: Chem.Mol) -> Chem.Mol:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        te = rdMolStandardize.TautomerEnumerator()
        return te.Canonicalize(mol)

    def _chemaxon_protonate_tautomerize(self, smilies: List[str], temp_dir: str) -> List[List[Chem.Mol]]:
        input_sdf = os.path.join(temp_dir, "input.sdf")
        protonated_sdf = os.path.join(temp_dir, "protonated.sdf")
        tautomer_sdf = os.path.join(temp_dir, "final.sdf")

        mols = []
        writer = Chem.SDWriter(input_sdf)
        for idx, smi in enumerate(smilies):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mol = Chem.AddHs(mol)
                mol.SetProp("_Name", f"mol{idx:05d}")
                writer.write(mol)
            mols.append(mol)
        writer.close()

        logger.debug("Starting ChemAxon protonation and tautomerization.")
        with open(os.path.join(temp_dir, "chemaxon_errors.log"), "a") as stderr_file:
            try:
                subprocess.run([
                    self.chemaxon_command, "-g", "majormicrospecies",
                    "-H", str(self.protonation_pH), "-f", "sdf", "-K", "true", input_sdf
                ], stdout=open(protonated_sdf, "w"), stderr=stderr_file, check=True, timeout=180)  # Adjusted timeout to 180 seconds (3 minutes)
            except subprocess.CalledProcessError as e:
                logger.error(f"ChemAxon command failed: {e}")
                raise
            except subprocess.TimeoutExpired as e:
                logger.error(f"ChemAxon command timed out: {e}")
                raise

            try:
                subprocess.run([
                    self.chemaxon_command, "-g", "canonicaltautomer",
                    "-C", "true", "-f", "sdf", protonated_sdf
                ], stdout=open(tautomer_sdf, "w"), stderr=stderr_file, check=True, timeout=180)  # Adjusted timeout to 180 seconds (3 minutes)
            except subprocess.CalledProcessError as e:
                logger.error(f"ChemAxon command failed: {e}")
                raise
            except subprocess.TimeoutExpired as e:
                logger.error(f"ChemAxon command timed out: {e}")
                raise

            final_sdf = tautomer_sdf

            if self.generate_conformers_chemaxon:
                logger.debug("Running ChemAxon conformer generation.")
                conf_sdf = os.path.join(temp_dir, "final_conformers.sdf")
                try:
                    subprocess.run([
                        self.chemaxon_command, "-g", "conformers", "-d", "0.2",
                        "-m", "50", "-O", "0", final_sdf
                    ], stdout=open(conf_sdf, "w"), stderr=stderr_file, check=True, timeout=180)  # Adjusted timeout to 180 seconds (3 minutes)
                except subprocess.CalledProcessError as e:
                    logger.error(f"ChemAxon command failed: {e}")
                    raise
                except subprocess.TimeoutExpired as e:
                    logger.error(f"ChemAxon command timed out: {e}")
                    raise
                final_sdf = conf_sdf

        logger.debug("ChemAxon processing complete. Reading final SDF.")

        name_to_mols = {}
        for mol in Chem.SDMolSupplier(final_sdf, removeHs=False):
            if mol:
                name = mol.GetProp("_Name")
                name_to_mols.setdefault(name, []).append(mol)

        result_mols = []
        for idx, original in enumerate(mols):
            name = f"mol{idx:05d}"
            result_mols.append(name_to_mols.get(name, []))

        return result_mols


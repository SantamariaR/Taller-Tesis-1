import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def find_unique_lineage_control_flexible(
    df_mergers,
    df_pool,
    match_cols=None,
    snap_weight=0.5,
    verbose=True,
):
    """
    Matching de galaxias con merger contra un pool de control sin merger.
    A diferencia de la versión original, NO restringe la búsqueda al mismo
    SnapNum, sino que busca en todo el pool y penaliza la distancia temporal.

    Parámetros
    ----------
    df_mergers : DataFrame
        Galaxias con merger. Debe contener 'RootGalaxyID', 'GalaxyID',
        'SnapNum' y las columnas en match_cols.
    df_pool : DataFrame
        Candidatas sin merger (evolución completa). Mismas columnas requeridas.
    match_cols : list[str], opcional
        Variables físicas para el matching. Por defecto:
        ['12+log(O/H)', 'Fgas', 'log_Mstar']
    snap_weight : float, opcional (default=0.5)
        Peso de la diferencia temporal en el espacio de distancias.
        0.0  → idéntico a buscar en todo el pool sin penalización temporal.
        0.5  → penalización moderada (recomendado como punto de partida).
        2.0+ → se aproxima al comportamiento de la versión original (mismo snap).
    verbose : bool
        Imprime advertencias cuando no hay candidatas disponibles.

    Retorna
    -------
    DataFrame con una fila por galaxia con merger, incluyendo las columnas
    de la gemela con prefijo 'control_', la distancia física y la diferencia
    de snaps entre la pareja.

    Notas
    -----
    - El escalamiento se realiza globalmente sobre todo el pool para
      consistencia entre snaps.
    - Cada RootGalaxyID del pool puede ser asignado como máximo una vez
      (matching sin reemplazo a nivel de linaje).
    - El espacio de distancias combinado es:
        d_total = sqrt(d_fisica² + (snap_weight * delta_snap_norm)²)
      donde delta_snap_norm = |snap_merger - snap_pool| / std(SnapNum_pool).
    """
    if match_cols is None:
        match_cols = ['12+log(O/H)', 'Fgas', 'log_Mstar']

    # ── Validaciones básicas ──────────────────────────────────────────────────
    required = match_cols + ['RootGalaxyID', 'GalaxyID', 'SnapNum']
    for col in required:
        if col not in df_mergers.columns:
            raise ValueError(f"df_mergers no tiene la columna '{col}'")
        if col not in df_pool.columns:
            raise ValueError(f"df_pool no tiene la columna '{col}'")

    # ── Copia de trabajo ──────────────────────────────────────────────────────
    df_mergers = (
        df_mergers.copy()
        .sort_values('SnapNum', ascending=True)
        .astype({'SnapNum': int, 'RootGalaxyID': int})
    )
    df_pool = df_pool.copy().astype({'SnapNum': int, 'RootGalaxyID': int})

    # ── Escalamiento global de variables físicas ──────────────────────────────
    scaler = StandardScaler()
    scaler.fit(df_pool[match_cols])

    pool_scaled_phys = scaler.transform(df_pool[match_cols])   # (N_pool, n_vars)

    # ── Normalización de SnapNum (para la penalización temporal) ──────────────
    snap_std = df_pool['SnapNum'].std()
    if snap_std == 0:
        snap_std = 1.0  # evitar división por cero si todos los snaps son iguales

    pool_snap_norm = (df_pool['SnapNum'].values - df_pool['SnapNum'].mean()) / snap_std

    # ── Construcción del espacio aumentado: [vars_físicas | snap_ponderado] ───
    # Columna temporal añadida al final con el peso solicitado
    pool_augmented = np.hstack([
        pool_scaled_phys,
        (pool_snap_norm * snap_weight).reshape(-1, 1)
    ])

    # ── Matching ──────────────────────────────────────────────────────────────
    used_pool_root_ids = set()
    results = []
    no_match_count = 0

    for idx, merger_row in df_mergers.iterrows():

        snap_merger = int(merger_row['SnapNum'])

        # Máscara: candidatas cuyos linajes aún no fueron usados
        available_mask = ~df_pool['RootGalaxyID'].isin(used_pool_root_ids)
        available_mask &= df_pool[match_cols].notna().all(axis=1)

        if available_mask.sum() == 0:
            if verbose:
                print(
                    f"[WARN] Sin candidatas disponibles para "
                    f"GalaxyID={merger_row['GalaxyID']} (Snap {snap_merger}). "
                    f"Se agotó el pool."
                )
            no_match_count += 1
            continue

        pool_avail_idx = df_pool.index[available_mask]
        pool_avail_aug = pool_augmented[available_mask.values]

        # Vector de la galaxia con merger en el espacio aumentado
        snap_merger_norm = (snap_merger - df_pool['SnapNum'].mean()) / snap_std
        merger_phys_scaled = scaler.transform(
            merger_row[match_cols].to_frame().T
        )
        merger_aug = np.hstack([
            merger_phys_scaled,
            np.array([[snap_merger_norm * snap_weight]])
        ])

        # Vecino más cercano en el espacio aumentado
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nn.fit(pool_avail_aug)
        distance, local_index = nn.kneighbors(merger_aug)

        matched_pool_row = df_pool.loc[pool_avail_idx[local_index[0][0]]]
        winning_root_id  = int(matched_pool_row['RootGalaxyID'])

        # Bloquear linaje
        used_pool_root_ids.add(winning_root_id)

        # Guardar resultado
        match_info = merger_row.to_dict()
        control_data = matched_pool_row.add_prefix('control_')
        match_info.update(control_data.to_dict())
        match_info['distancia_matching']    = float(distance[0][0])
        match_info['distancia_fisica']      = float(
            np.linalg.norm(merger_phys_scaled - scaler.transform(
                matched_pool_row[match_cols].to_frame().T
            ))
        )
        match_info['delta_snap_matching']   = abs(snap_merger - int(matched_pool_row['SnapNum']))
        results.append(match_info)

    if verbose:
        n_matched = len(results)
        n_total   = len(df_mergers)
        print(
            f"\nMatching completado: {n_matched}/{n_total} galaxias emparejadas "
            f"({no_match_count} sin pareja por pool agotado)."
        )
        if results:
            df_res = pd.DataFrame(results)
            print(f"  Distancia física mediana:   {df_res['distancia_fisica'].median():.4f}")
            print(f"  Delta snap mediano:         {df_res['delta_snap_matching'].median():.1f}")
            print(f"  Pares con mismo snap:       {(df_res['delta_snap_matching'] == 0).sum()} "
                  f"({(df_res['delta_snap_matching'] == 0).mean()*100:.1f}%)")

    return pd.DataFrame(results)


# ─── Función auxiliar: diagnóstico del matching ───────────────────────────────

def diagnostico_matching(df_matched, match_cols=None, figsize=(14, 4)):
    """
    Grafica histogramas de las diferencias entre cada pareja (merger - control)
    para evaluar la calidad del matching.

    Parámetros
    ----------
    df_matched : DataFrame
        Salida de find_unique_lineage_control_flexible.
    match_cols : list[str], opcional
        Variables a diagnosticar. Por defecto las 3 de matching + delta_snap.
    """
    import matplotlib.pyplot as plt

    if match_cols is None:
        match_cols = ['12+log(O/H)', 'Fgas', 'log_Mstar']

    diag_cols = [c for c in match_cols if c in df_matched.columns
                 and f'control_{c}' in df_matched.columns]

    n = len(diag_cols) + 1  # +1 para delta_snap
    fig, axes = plt.subplots(1, n, figsize=figsize)
    fig.suptitle("Diagnóstico del matching — diferencias pareja (merger − control)",
                 fontweight='bold')

    for ax, col in zip(axes[:-1], diag_cols):
        diff = df_matched[col] - df_matched[f'control_{col}']
        ax.hist(diff.dropna(), bins=30, color='#457B9D', edgecolor='white', alpha=0.85)
        ax.axvline(0,   color='black', linewidth=1.5, linestyle='--')
        ax.axvline(diff.median(), color='#E63946', linewidth=1.5,
                   linestyle='-', label=f'Mediana={diff.median():.3f}')
        ax.set_title(f'Δ {col}')
        ax.set_xlabel('Diferencia')
        ax.set_ylabel('Frecuencia')
        ax.legend(fontsize=9)

    # Delta snap
    ax = axes[-1]
    ax.hist(df_matched['delta_snap_matching'], bins=20,
            color='#2A9D8F', edgecolor='white', alpha=0.85)
    ax.axvline(df_matched['delta_snap_matching'].median(), color='#E63946',
               linewidth=1.5, label=f"Mediana={df_matched['delta_snap_matching'].median():.1f}")
    ax.set_title('Δ SnapNum entre pares')
    ax.set_xlabel('|snap_merger − snap_control|')
    ax.set_ylabel('Frecuencia')
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig

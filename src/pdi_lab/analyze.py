from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .image_io import read_image, write_image


def _write_components_csv(path: Path, stats: np.ndarray, centroids: np.ndarray, min_area: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_count = 0
    kept_count = 0
    raw_area = 0
    kept_area = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "area", "x", "y", "width", "height", "cx", "cy", "kept"])
        # stats[0] corresponde ao fundo e não é exportado como objeto.
        for label in range(1, len(stats)):
            x, y, width, height, area = (int(v) for v in stats[label])
            kept = area >= min_area
            cx, cy = (float(v) for v in centroids[label])
            writer.writerow([label, area, x, y, width, height, f"{cx:.3f}", f"{cy:.3f}", str(kept).lower()])
            raw_count += 1
            raw_area += area
            if kept:
                kept_count += 1
                kept_area += area
    return {"raw_count": raw_count, "filtered_count": kept_count,
            "raw_foreground_area": raw_area, "filtered_foreground_area": kept_area}


def _connected_components(mask: np.ndarray, connectivity: int):
    # cv2.connectedComponentsWithStats fornece rótulos, área, caixa e centroide.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity, ltype=cv2.CV_32S
    )
    # count inclui o fundo; stats e centroids têm o mesmo número de entradas.
    return labels, stats, centroids, int(count - 1)


def _label_visualization(mask: np.ndarray, connectivity: int, min_area: int) -> np.ndarray:
    labels, stats, centroids, _ = _connected_components(mask, connectivity)
    height, width = mask.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for label in range(1, len(stats)):
        x, y, box_w, box_h, area = (int(v) for v in stats[label])
        if area < min_area:
            continue
        # Cor determinística para facilitar comparação visual dos rótulos.
        color = (int((37 * label + 70) % 256), int((97 * label + 130) % 256), int((173 * label + 40) % 256))
        canvas[labels == label] = color
        cv2.rectangle(canvas, (x, y), (x + box_w - 1, y + box_h - 1), (255, 255, 255), 1)
        cx, cy = centroids[label]
        cv2.circle(canvas, (int(round(cx)), int(round(cy))), 2, (255, 255, 255), -1)
        cv2.putText(canvas, str(label), (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def analyze_mask(input_path: str, output_prefix: str, connectivity: int = 8,
                 min_area: int = 0, kernel_shape: str = "ellipse",
                 kernel_size: int = 3, open_iterations: int = 1,
                 close_iterations: int = 1) -> dict:
    if connectivity not in (4, 8):
        raise ValueError("--connectivity deve ser 4 ou 8.")
    if min_area < 0:
        raise ValueError("--min-area deve ser inteiro nao negativo.")
    if kernel_shape not in {"rect", "ellipse", "cross"}:
        raise ValueError("--kernel-shape deve ser rect, ellipse ou cross.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("--kernel-size deve ser inteiro impar e >= 1.")
    if open_iterations < 0 or close_iterations < 0 or (open_iterations == 0 and close_iterations == 0):
        raise ValueError("--open-iterations e --close-iterations devem ser >= 0 e pelo menos uma operacao deve ocorrer.")

    source = read_image(input_path)
    if source.ndim != 2:
        raise ValueError("A entrada de analyze deve ser uma mascara binaria de um canal.")
    unique = set(int(v) for v in np.unique(source))
    if not unique.issubset({0, 255}):
        raise ValueError(f"A entrada deve conter somente 0 e 255; encontrados: {sorted(unique)}")
    initial = source.astype(np.uint8, copy=True)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = prefix.name

    # Guardar uma cópia explícita da máscara original recebida.
    write_image(str(prefix.parent / f"{stem}_initial.png"), initial)
    labels4, stats4, centroids4, count4 = _connected_components(initial, 4)
    labels8, stats8, centroids8, count8 = _connected_components(initial, 8)
    selected_initial_stats = stats4 if connectivity == 4 else stats8
    selected_initial_centroids = centroids4 if connectivity == 4 else centroids8
    init_csv = _write_components_csv(prefix.parent / f"{stem}_initial.csv", selected_initial_stats, selected_initial_centroids, min_area)

    shape_code = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}[kernel_shape]
    kernel = cv2.getStructuringElement(shape_code, (kernel_size, kernel_size))
    current = initial.copy()
    stages = []
    if open_iterations > 0:
        current = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
        current = np.where(current > 0, 255, 0).astype(np.uint8)
        stage_path = prefix.parent / f"{stem}_opened.png"
        write_image(str(stage_path), current)
        stages.append({"operation": "opening", "iterations": open_iterations, "file": stage_path.name,
                       "foreground_pixels": int(np.count_nonzero(current))})
    if close_iterations > 0:
        current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
        current = np.where(current > 0, 255, 0).astype(np.uint8)
        stage_path = prefix.parent / f"{stem}_final.png"
        write_image(str(stage_path), current)
        stages.append({"operation": "closing", "iterations": close_iterations, "file": stage_path.name,
                       "foreground_pixels": int(np.count_nonzero(current))})
    # Se a última operação foi abertura (fechamento desligado), ela também é a saída final.
    final_path = prefix.parent / f"{stem}_final.png"
    if not final_path.exists():
        write_image(str(final_path), current)

    final_labels, final_stats, final_centroids, final_raw_count = _connected_components(current, connectivity)
    final_csv = _write_components_csv(prefix.parent / f"{stem}_final.csv", final_stats, final_centroids, min_area)
    vis = _label_visualization(current, connectivity, min_area)
    labels_path = prefix.parent / f"{stem}_labels.png"
    write_image(str(labels_path), vis)

    initial_pixels = int(np.count_nonzero(initial))
    final_pixels = int(np.count_nonzero(current))
    summary = {
        "input": str(input_path), "output_prefix": str(prefix),
        "dimensions": {"width": int(initial.shape[1]), "height": int(initial.shape[0])},
        "mask_contract": {"single_channel": True, "binary_values": [0, 255], "foreground_is": 255},
        "connectivity": connectivity,
        "initial_connectivity_counts": {"4": count4, "8": count8},
        "min_area": min_area,
        "morphology": {"order": [s["operation"] for s in stages], "kernel_shape": kernel_shape,
                        "kernel_size": [kernel_size, kernel_size], "open_iterations": open_iterations,
                        "close_iterations": close_iterations, "stages": stages},
        "initial": {**init_csv, "foreground_pixels": initial_pixels,
                    "area_sum_matches_foreground_pixels": init_csv["raw_foreground_area"] == initial_pixels},
        "final": {**final_csv, "foreground_pixels": final_pixels,
                  "raw_component_count": final_raw_count,
                  "area_sum_matches_foreground_pixels": final_csv["raw_foreground_area"] == final_pixels},
        "artifacts": {"initial_mask": f"{stem}_initial.png", "initial_csv": f"{stem}_initial.csv",
                      "final_mask": f"{stem}_final.png", "final_csv": f"{stem}_final.csv",
                      "labels_visualization": labels_path.name}
    }
    summary_path = prefix.parent / f"{stem}_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Componentes iniciais: conectividade 4={count4}; conectividade 8={count8}")
    print(f"Componentes finais ({connectivity}-conexa): brutos={final_csv['raw_count']}; filtrados={final_csv['filtered_count']}")
    print(f"Pixels de primeiro plano: inicial={initial_pixels}; final={final_pixels}")
    print(f"Resumo: {summary_path}")
    return summary

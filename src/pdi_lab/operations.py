from __future__ import annotations
from .image_io import read_image, write_image
import numpy as np
import sys
import csv
import cv2
from .kernel import read_kernel
from . import exit_code
from .cli import CliOptions
#O codigo a seguir inclui todas as operacoes dos labs m1-1-2-3.
#Antes de rodar acesse a pasta python (nos meus testes eu precisava fazer cd pdi_template_main, depois cd python para chegar na pasta certa)
#Para rodar o codigo inserir o comando desejado, exemplo: 
#python -m src.pdi_lab --operation grayscale_weighted --input images/input/m1_color_2x2.png --output images/output/gray_weighted.png

# A lista é equivalente às variantes C++ e Java. Atividades complementares
# dos roteiros não são transformadas em requisitos do contrato mínimo.
KNOWN_OPERATIONS = {
    # M1.1 — representação, canais e níveis de cinza FEITO
    "inspect",
    "copy",
    "channel_b",
    "channel_g",
    "channel_r",
    "grayscale_average",
    "grayscale_weighted",
    "quantize",

    # M1.2 — transformações de intensidade FEITO
    "brightness",
    "contrast",
    "negative",
    "threshold",
    "histogram",

    # M1.3 — convolução e filtragem espacial FEITO
    "convolution",
    "mean_filter",
    "weighted_mean",
    "laplacian",
    "sobel",
    # M2.1 — segmentação binária
    "segment",
    # M2.2 — morfologia e componentes conexos
    "analyze",
}


def is_known_operation(operation: str) -> bool:
    """Retorna True somente para as operações previstas no contrato comum."""

    return operation in KNOWN_OPERATIONS
def brightness(image, value):
    height = image.shape[0]
    width = image.shape[1]

    result = np.zeros_like(image)

    if len(image.shape) == 2:
        for y in range(height):
            for x in range(width):
                new_value = int(image[y, x]) + value

                if new_value > 255:
                    new_value = 255

                if new_value < 0:
                    new_value = 0

                result[y, x] = new_value

    elif len(image.shape) == 3:
        channels = image.shape[2]

        for y in range(height):
            for x in range(width):
                for channel in range(channels):
                    new_value = int(image[y, x, channel]) + value

                    if new_value > 255:
                        new_value = 255

                    if new_value < 0:
                        new_value = 0

                    result[y, x, channel] = new_value

    else:
        raise ValueError("Formato de imagem não suportado.")

    return result
def contrast(image, alpha):
    height = image.shape[0]
    width = image.shape[1]

    result = np.zeros_like(image)

    if len(image.shape) == 2:
        for y in range(height):
            for x in range(width):
                value = int(image[y, x])

                new_value = alpha * (value - 128) + 128
                new_value = int(round(new_value))

                if new_value > 255:
                    new_value = 255

                if new_value < 0:
                    new_value = 0

                result[y, x] = new_value

    elif len(image.shape) == 3:
        channels = image.shape[2]

        for y in range(height):
            for x in range(width):
                for channel in range(channels):
                    value = int(image[y, x, channel])

                    new_value = alpha * (value - 128) + 128
                    new_value = int(round(new_value))

                    if new_value > 255:
                        new_value = 255

                    if new_value < 0:
                        new_value = 0

                    result[y, x, channel] = new_value

    else:
        raise ValueError("Formato de imagem não suportado.")

    return result
def negative(image):
    height = image.shape[0]
    width = image.shape[1]

    result = np.zeros_like(image)

    if len(image.shape) == 2:
        for y in range(height):
            for x in range(width):
                result[y, x] = 255 - int(image[y, x])

    elif len(image.shape) == 3:
        channels = image.shape[2]

        for y in range(height):
            for x in range(width):
                for channel in range(channels):
                    result[y, x, channel] = 255 - int(
                        image[y, x, channel]
                    )

    else:
        raise ValueError("Formato de imagem não suportado.")

    return result
def threshold(image, value):
    height = image.shape[0]
    width = image.shape[1]

    result = np.zeros_like(image)

    if len(image.shape) == 2:
        for y in range(height):
            for x in range(width):
                pixel = int(image[y, x])

                if pixel >= value:
                    result[y, x] = 255
                else:
                    result[y, x] = 0

    elif len(image.shape) == 3:
        channels = image.shape[2]

        for y in range(height):
            for x in range(width):
                for channel in range(channels):
                    pixel = int(image[y, x, channel])

                    if pixel >= value:
                        result[y, x, channel] = 255
                    else:
                        result[y, x, channel] = 0

    else:
        raise ValueError("Formato de imagem não suportado.")

    return result
def histogram(image, output):
    height = image.shape[0]
    width = image.shape[1]

    if len(image.shape) != 2:
        raise ValueError("O histograma deve receber uma imagem em tons de cinza.")

    counts = [0] * 256

    for y in range(height):
        for x in range(width):
            value = int(image[y, x])
            counts[value] += 1

    with open(output, "w", encoding="utf-8") as file:
        file.write("intensidade,quantidade\n")

        for intensity in range(256):
            file.write(
                f"{intensity},{counts[intensity]}\n"
            )

def grayscale_weighted(image):
    height = image.shape[0]
    width = image.shape[1]

    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            B = int(image[y, x, 0])
            G = int(image[y, x, 1])
            R = int(image[y, x, 2])

            gray = 0.299 * R + 0.587 * G + 0.114 * B

            result[y, x] = int(round(gray))

    return result
def quantize(image, levels):
    if levels <= 0:
        raise ValueError("A quantidade de níveis deve ser positiva.")

    height = image.shape[0]
    width = image.shape[1]

    result = np.zeros_like(image)

    step = 256 / levels

    for y in range(height):
        for x in range(width):
            value = int(image[y, x])

            level = int(value / step)

            if level >= levels:
                level = levels - 1

            quantized = int(round(level * 255 / (levels - 1)))

            result[y, x] = quantized

    return result
def separate_channel(image, channel):
    height = image.shape[0]
    width = image.shape[1]

    result = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            result[y, x, channel] = image[y, x, channel]

    return result
def grayscale_average(image):
    height = image.shape[0]
    width = image.shape[1]

    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            B = int(image[y, x, 0])
            G = int(image[y, x, 1])
            R = int(image[y, x, 2])

            gray = (R + G + B) / 3

            result[y, x] = int(round(gray))

    return result


def get_pixel(image, y, x, border):
    height = image.shape[0]
    width = image.shape[1]

    
    if 0 <= y < height and 0 <= x < width:
        return image[y, x]

    
    if border == "replicate":
        y = max(0, min(y, height - 1))
        x = max(0, min(x, width - 1))

        return image[y, x]

   
    if border == "copy":
        return None

    raise ValueError(
        f"Estrategia de borda desconhecida: {border}"
    )


def convolution(image, kernel, border="replicate"):
    height = image.shape[0]
    width = image.shape[1]

    result = np.zeros_like(image)

    radius = kernel.radius


    if len(image.shape) == 2:

        for y in range(height):
            for x in range(width):

                total = 0.0

                for ky in range(kernel.size):
                    for kx in range(kernel.size):

                        source_y = y + ky - radius
                        source_x = x + kx - radius

                        pixel = get_pixel(
                            image,
                            source_y,
                            source_x,
                            border
                        )

                        # Na estratégia COPY, posições fora
                        # da imagem não participam da soma.
                        if pixel is not None:
                            coefficient = kernel.at(ky, kx)

                            total += float(pixel) * coefficient

                value = int(round(total))

                if value > 255:
                    value = 255

                if value < 0:
                    value = 0

                result[y, x] = value

    

    elif len(image.shape) == 3:

        channels = image.shape[2]

        for y in range(height):
            for x in range(width):

                for channel in range(channels):

                    total = 0.0

                    for ky in range(kernel.size):
                        for kx in range(kernel.size):

                            source_y = y + ky - radius
                            source_x = x + kx - radius

                            pixel = get_pixel(
                                image,
                                source_y,
                                source_x,
                                border
                            )

                            if pixel is not None:
                                coefficient = kernel.at(ky, kx)

                                total += (
                                    float(pixel[channel])
                                    * coefficient
                                )

                    value = int(round(total))

                    if value > 255:
                        value = 255

                    if value < 0:
                        value = 0

                    result[y, x, channel] = value

    else:
        raise ValueError(
            "Formato de imagem não suportado."
        )

    return result


def mean_filter(image, size, border="replicate"):
    if size <= 0 or size % 2 == 0:
        raise ValueError(
            "O tamanho do filtro deve ser positivo e ímpar."
        )

    kernel = np.ones(
        (size, size),
        dtype=np.float64
    )

    kernel = kernel / (size * size)

    
    values = tuple(
        float(kernel[y, x])
        for y in range(size)
        for x in range(size)
    )

    from .kernel import Kernel

    kernel_object = Kernel(
        size=size,
        values=values
    )

    return convolution(
        image,
        kernel_object,
        border
    )


def weighted_mean(image, border="replicate"):
    values = (
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0
    )

    values = tuple(value / 16.0 for value in values)

    from .kernel import Kernel

    kernel = Kernel(
        size=3,
        values=values
    )

    return convolution(
        image,
        kernel,
        border
    )


def laplacian(image, border="replicate"):
    values = (
         0.0, -1.0,  0.0,
        -1.0,  4.0, -1.0,
         0.0, -1.0,  0.0
    )

    from .kernel import Kernel

    kernel = Kernel(
        size=3,
        values=values
    )

    return convolution(
        image,
        kernel,
        border
    )


def sobel(image, border="replicate"):
    if len(image.shape) != 2:
        raise ValueError(
            "O operador Sobel deve receber uma imagem "
            "em tons de cinza."
        )

    kernel_x = (
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
    )

    kernel_y = (
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0
    )

    height = image.shape[0]
    width = image.shape[1]

    result = np.zeros_like(image)

    for y in range(height):
        for x in range(width):

            gx = 0.0
            gy = 0.0

            for ky in range(3):
                for kx in range(3):

                    source_y = y + ky - 1
                    source_x = x + kx - 1

                    pixel = get_pixel(
                        image,
                        source_y,
                        source_x,
                        border
                    )

                    if pixel is None:
                        continue

                    coefficient_x = kernel_x[ky * 3 + kx]
                    coefficient_y = kernel_y[ky * 3 + kx]

                    pixel = float(pixel)

                    gx += pixel * coefficient_x
                    gy += pixel * coefficient_y

            magnitude = (gx * gx + gy * gy) ** 0.5

            value = int(round(magnitude))

            if value > 255:
                value = 255

            if value < 0:
                value = 0

            result[y, x] = value

    return result

def segment_image(image, method, threshold_value=120, adaptive_block=11,
                  adaptive_c=2, hsv_lower=(0, 0, 0), hsv_upper=(179, 255, 255),
                  invert=False):
    """Gera máscara binária uint8 (0=fundo, 255=objeto) para um método."""
    if image is None or image.size == 0:
        raise ValueError("Imagem de entrada vazia.")
    if method not in {"fixed", "otsu", "adaptive", "hsv"}:
        raise ValueError(f"Método de segmentação desconhecido: {method}")

    if method == "hsv":
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("O método hsv exige uma imagem colorida BGR.")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = tuple(int(v) for v in hsv_lower)
        upper = tuple(int(v) for v in hsv_upper)
        if lower[0] <= upper[0]:
            mask = cv2.inRange(hsv, lower, upper)
        else:
            # Intervalo de matiz cruza a descontinuidade 179 -> 0.
            low_a = cv2.inRange(hsv, (0, lower[1], lower[2]), upper)
            high_b = cv2.inRange(hsv, lower, (179, upper[1], upper[2]))
            mask = cv2.bitwise_or(low_a, high_b)
    else:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if method == "fixed":
            if not 0 <= int(threshold_value) <= 255:
                raise ValueError("--threshold deve estar entre 0 e 255.")
            # Contrato deste projeto: pixels iguais ao limiar entram no objeto.
            mask = np.where(gray >= int(threshold_value), 255, 0).astype(np.uint8)
        elif method == "otsu":
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            if adaptive_block < 3 or adaptive_block % 2 == 0:
                raise ValueError("--adaptive_block deve ser ímpar e >= 3.")
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, int(adaptive_block), float(adaptive_c)
            )

    if invert:
        mask = cv2.bitwise_not(mask)
    # Garante contrato de máscara de um canal com valores {0,255}.
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _parse_triplet(text, parameter_name):
    try:
        values = tuple(int(v.strip()) for v in text.split(","))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"--{parameter_name} deve ter três inteiros separados por vírgula.") from error
    if len(values) != 3:
        raise ValueError(f"--{parameter_name} deve ter três inteiros separados por vírgula.")
    limits = (179, 255, 255)
    if any(v < 0 or v > limits[i] for i, v in enumerate(values)):
        raise ValueError(f"--{parameter_name}: H deve estar em 0..179 e S/V em 0..255.")
    return values


def run_operation(options: CliOptions) -> int:
    if options.operation == "analyze":
        from .analyze import analyze_mask
        params = options.parameters
        analyze_mask(
            options.input, options.output,
            connectivity=int(params.get("connectivity", "8")),
            min_area=int(params.get("min_area", "0")),
            kernel_shape=params.get("kernel_shape", "ellipse"),
            kernel_size=int(params.get("kernel_size", "3")),
            open_iterations=int(params.get("open_iterations", "1")),
            close_iterations=int(params.get("close_iterations", "1")),
        )
        return exit_code.SUCCESS
    if options.operation == "segment":
        image = read_image(options.input)
        params = options.parameters
        method = params.get("method", "fixed")
        lower = _parse_triplet(params.get("hsv_lower", "0,100,100"), "hsv_lower")
        upper = _parse_triplet(params.get("hsv_upper", "10,255,255"), "hsv_upper")
        invert = params.get("invert", "false").lower() in {"1", "true", "yes"}
        mask = segment_image(
            image, method,
            threshold_value=int(params.get("threshold", "120")),
            adaptive_block=int(params.get("adaptive_block", "11")),
            adaptive_c=float(params.get("adaptive_c", "2")),
            hsv_lower=lower, hsv_upper=upper, invert=invert,
        )
        write_image(options.output, mask)
        foreground = int(np.count_nonzero(mask == 255))
        total = int(mask.size)
        print(f"Metodo: {method}")
        print(f"Dimensoes: {mask.shape[1]}x{mask.shape[0]}")
        print(f"Pixels de primeiro plano: {foreground}/{total} ({100*foreground/total:.2f}%)")
        print(f"Valores da mascara: {sorted(int(v) for v in np.unique(mask))}")
        return exit_code.SUCCESS
    if options.operation == "copy":
        image = read_image(options.input)
        write_image(options.output, image)
        return 0
    if options.operation == "inspect":
        image = read_image(options.input)
        height = image.shape[0]
        width = image.shape[1]

        if len(image.shape) == 2:
            channels = 1
        else:
            channels = image.shape[2]

        print(f"Largura: {width}")
        print(f"Altura: {height}")
        print(f"Canais: {channels}")
        print(f"Tipo: {image.dtype}")
        print(f"Quantidade de pixels: {width * height}")

        if channels == 1:
            minimum = 255
            maximum = 0
            total = 0

            for y in range(height):
                for x in range(width):
                    value = int(image[y, x])

                    if value < minimum:
                        minimum = value

                    if value > maximum:
                        maximum = value

                    total += value

            average = total / (width * height)

            print(f"Mínimo: {minimum}")
            print(f"Máximo: {maximum}")
            print(f"Média: {average:.2f}")

        elif channels == 3:
            minimum = [255, 255, 255]
            maximum = [0, 0, 0]
            total = [0, 0, 0]

            for y in range(height):
                for x in range(width):
                    for channel in range(3):
                        value = int(image[y, x, channel])

                        if value < minimum[channel]:
                            minimum[channel] = value

                        if value > maximum[channel]:
                            maximum[channel] = value

                        total[channel] += value

            pixels = width * height

            print(f"Mínimo B: {minimum[0]}")
            print(f"Máximo B: {maximum[0]}")
            print(f"Média B: {total[0] / pixels:.2f}")

            print(f"Mínimo G: {minimum[1]}")
            print(f"Máximo G: {maximum[1]}")
            print(f"Média G: {total[1] / pixels:.2f}")

            print(f"Mínimo R: {minimum[2]}")
            print(f"Máximo R: {maximum[2]}")
            print(f"Média R: {total[2] / pixels:.2f}")

        else:
            raise ValueError("Número de canais não suportado.")
        return 0
    
    if options.operation == "channel_b":
        image = read_image(options.input)
        result = separate_channel(image, 0)
        write_image(options.output, result)
        return 0

    if options.operation == "channel_g":
        image = read_image(options.input)
        result = separate_channel(image, 1)
        write_image(options.output, result)
        return 0

    if options.operation == "channel_r":
        image = read_image(options.input)
        result = separate_channel(image, 2)
        write_image(options.output, result)
        return 0
    if options.operation == "grayscale_average":
        image = read_image(options.input)
        result = grayscale_average(image)
        write_image(options.output, result)
        return 0
    if options.operation == "grayscale_weighted":
        image = read_image(options.input)
        result = grayscale_weighted(image)
        write_image(options.output, result)
        return 0
    if options.operation == "quantize":
        image = read_image(options.input)

        levels = int(options.parameters["levels"])

        result = quantize(image, levels)

        write_image(options.output, result)
        return 0
    if options.operation == "brightness":
        image = read_image(options.input)

        value = int(options.parameters["value"])

        result = brightness(image, value)

        write_image(options.output, result)
        return 0
    if options.operation == "contrast":
        image = read_image(options.input)

        alpha = float(options.parameters["alpha"])

        result = contrast(image, alpha)

        write_image(options.output, result)
        return 0
    if options.operation == "negative":
        image = read_image(options.input)

        result = negative(image)

        write_image(options.output, result)
        return 0
    if options.operation == "threshold":
        image = read_image(options.input)

        value = int(options.parameters["threshold"])

        result = threshold(image, value)

        write_image(options.output, result)
        return 0
    if options.operation == "histogram":
        image = read_image(options.input)

        histogram(image, options.output)

        return 0

    if options.operation == "convolution":
        image = read_image(options.input)

        kernel = read_kernel(
            options.parameters["kernel"]
        )

        border = options.parameters.get(
            "border",
            "replicate"
        )

        result = convolution(
            image,
            kernel,
            border
        )

        write_image(
            options.output,
            result
        )

        return 0


    if options.operation == "mean_filter":
        image = read_image(options.input)

        size = int(
            options.parameters.get(
                "size",
                "3"
            )
        )

        border = options.parameters.get(
            "border",
            "replicate"
        )

        result = mean_filter(
            image,
            size,
            border
        )

        write_image(
            options.output,
            result
        )

        return 0


    if options.operation == "weighted_mean":
        image = read_image(options.input)

        border = options.parameters.get(
            "border",
            "replicate"
        )

        result = weighted_mean(
            image,
            border
        )

        write_image(
            options.output,
            result
        )

        return 0


    if options.operation == "laplacian":
        image = read_image(options.input)

        border = options.parameters.get(
            "border",
            "replicate"
        )

        result = laplacian(
            image,
            border
        )

        write_image(
            options.output,
            result
        )

        return 0


    if options.operation == "sobel":
        image = read_image(options.input)

        border = options.parameters.get(
            "border",
            "replicate"
        )

        result = sobel(
            image,
            border
        )

        write_image(
            options.output,
            result
        )

        return 0
    print(
        f"Operacao '{options.operation}' reconhecida, "
        "mas ainda nao implementada no projeto-base.",
        file=sys.stderr,
    )
    return exit_code.GENERAL_ERROR
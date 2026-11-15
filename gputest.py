"""CUDA GPU의 실제 행렬 곱셈 처리량(TFLOPS)을 측정한다.

실행 예:
    python gputest.py
    python gputest.py --dtype fp16 --size 8192
    python gputest.py --device 1 --repeat 100

CUDA를 지원하는 PyTorch가 필요하다. 데이터 전송과 메모리 할당은
측정에서 제외하며, 결과는 선택한 자료형의 실측 성능이다.
"""

import argparse
import statistics
import sys


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("양의 정수를 입력하세요.")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=positive_int, default=4096,
                        help="정사각 행렬의 한 변 크기 (기본: 4096)")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"),
                        default="fp32", help="자료형 (기본: fp32, TF32 비활성화)")
    parser.add_argument("--warmup", type=positive_int, default=10,
                        help="워밍업 횟수 (기본: 10)")
    parser.add_argument("--repeat", type=positive_int, default=50,
                        help="각 측정 구간의 곱셈 반복 횟수 (기본: 50)")
    parser.add_argument("--trials", type=positive_int, default=5,
                        help="측정 구간 수 (기본: 5)")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA GPU 번호 (기본: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import torch
    except ImportError:
        print("오류: PyTorch가 없습니다. CUDA 지원 PyTorch를 설치한 뒤 실행하세요.\n"
              "설치 안내: https://pytorch.org/get-started/locally/", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("오류: CUDA GPU를 사용할 수 없습니다. GPU 드라이버와 CUDA 지원 "
              "PyTorch 설치를 확인하세요.", file=sys.stderr)
        return 1
    if not 0 <= args.device < torch.cuda.device_count():
        print(f"오류: GPU 번호는 0~{torch.cuda.device_count() - 1} 범위입니다.",
              file=sys.stderr)
        return 1

    torch.cuda.set_device(args.device)
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        print("오류: 선택한 GPU에서 BF16을 지원하지 않습니다.", file=sys.stderr)
        return 1

    # FP32 측정이 TF32로 대체되지 않도록 설정한다.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]
    props = torch.cuda.get_device_properties(args.device)
    print(f"GPU: {props.name} (cuda:{args.device})")
    print(f"PyTorch: {torch.__version__} | CUDA/HIP runtime: "
          f"{torch.version.cuda or torch.version.hip}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GiB")
    print(f"행렬: {args.size} × {args.size} | 자료형: {args.dtype}")
    if args.dtype == "fp32":
        print("FP32 정밀도: TF32 비활성화")
    print(f"워밍업: {args.warmup}회 | 측정: {args.repeat}회 × {args.trials}구간",
          flush=True)

    try:
        with torch.inference_mode():
            a = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
            b = torch.randn_like(a)
            out = torch.empty_like(a)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            for _ in range(args.warmup):
                torch.mm(a, b, out=out)
            torch.cuda.synchronize()

            # CUDA 이벤트를 미리 초기화한다.
            start.record()
            end.record()
            end.synchronize()

            times_ms = []
            operations = 2 * args.size**3
            for trial in range(args.trials):
                start.record()
                for _ in range(args.repeat):
                    torch.mm(a, b, out=out)
                end.record()
                end.synchronize()
                ms = start.elapsed_time(end) / args.repeat
                times_ms.append(ms)
                tflops = operations / (ms / 1000) / 1e12
                print(f"  구간 {trial + 1}: {ms:.3f} ms/회 | {tflops:.2f} TFLOPS")
    except torch.cuda.OutOfMemoryError:
        print("오류: GPU 메모리가 부족합니다. --size 2048 등으로 줄여 실행하세요.",
              file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"GPU 연산 오류: {exc}", file=sys.stderr)
        return 1

    median_ms = statistics.median(times_ms)
    print(f"\n중앙값: {median_ms:.3f} ms/회 | "
          f"{operations / (median_ms / 1000) / 1e12:.2f} TFLOPS")
    print(f"최고값: {operations / (min(times_ms) / 1000) / 1e12:.2f} TFLOPS")
    print("계산식: 2 × N³ ÷ 실행 시간(초) ÷ 10¹²")
    print("데이터 전송·메모리 할당 제외. 곱셈+덧셈을 2회 연산으로 계산합니다.")
    print("제품 사양과 비교할 때는 자료형과 희소성 적용 여부를 확인하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

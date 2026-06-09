# Exporter definition
"""
Exports a trained StandaloneModel to CoreML or ONNX format.

CoreML (.mlpackage) — best for Apple hardware deployment.
  ComputeUnit.ALL lets the runtime pick CPU / GPU / Neural Engine.

ONNX (.onnx) — portable, readable by TensorRT, OpenVINO, TFLite, etc.
  Use opset 18 (widely supported as of PyTorch 2.x).
"""

import torch


class ModelExporter:

    def export_coreml(
        self,
        model,
        input_shape=(1, 3, 32, 32),
        path='model.mlpackage',
        model_name='NASModel',
    ):
        """
        Export to CoreML for deployment on Apple devices (M4 Air).

        Requires: pip install coremltools

        ComputeUnit.ALL lets CoreML decide at runtime whether to use
        the CPU, GPU, or Neural Engine — important for M4 performance.
        """
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError(
                "coremltools not installed. Run: pip install coremltools"
            )

        model.eval()
        example = torch.zeros(input_shape)

        print("Tracing model for CoreML export...")
        traced = torch.jit.trace(model, example)

        print("Converting to CoreML...")
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name='input', shape=input_shape)],
            outputs=[ct.TensorType(name='logits')],
            compute_units=ct.ComputeUnit.ALL,
        )

        mlmodel.short_description = model_name
        mlmodel.save(path)
        print(f"CoreML model saved -> {path}")
        print(f"  Input  : {input_shape}")
        print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")
        return mlmodel

    def export_onnx(
        self,
        model,
        input_shape=(1, 3, 32, 32),
        path='model.onnx',
        opset=18,
    ):
        """
        Export to ONNX format for cross-platform deployment.

        opset=18 is recommended with PyTorch 2.x and onnxscript.
        """
        try:
            import onnx
        except ImportError:
            raise ImportError(
                "onnx is not installed. Run: pip install onnx"
            )

        model.eval()
        example = torch.zeros(input_shape)

        print(f"Exporting to ONNX (opset {opset})...")
        torch.onnx.export(
            model,
            example,
            path,
            opset_version=opset,
            input_names=['input'],
            output_names=['logits'],
            do_constant_folding=True,
        )
        print(f"ONNX model saved -> {path}")
        print(f"  Opset  : {opset}")
        print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")

    def print_model_info(self, model):
        """Print a summary of the standalone model's size and op distribution."""
        from collections import Counter
        from nas_rl.search_space.ops import OPS

        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)

        print(f"\nModel Summary")
        print(f"  Total parameters     : {total:,}  ({total/1e6:.3f}M)")
        print(f"  Trainable parameters : {trainable:,}")
        print(f"  Cells                : {model.n_cells}")
        print(f"  Nodes per cell       : {model.n_nodes}")

        # Infer op names by matching class types to OPS registry
        op_counts = Counter()
        for cell in model.cells:
            for key, op_module in cell.ops.items():
                op_counts[type(op_module).__name__] += 1

        print(f"  Op distribution:")
        for op_type, count in op_counts.most_common():
            print(f"    {op_type:30s}: {count}")
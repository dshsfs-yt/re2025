from transformers import Seq2SeqTrainer
import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Any, Tuple
from transformers.generation import LogitsProcessor, LogitsProcessorList
import torch

class RestrictVocabLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = torch.tensor(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask


class MonitoringSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer with enhanced monitoring capabilities
    """
    
    def __init__(self, *args, restrict_decode_vocab=None, **kwargs):
        self.restrict_decode_vocab = restrict_decode_vocab
        super().__init__(*args, **kwargs)
        self.step_count = 0
        self.print_every_n_steps = 25  # 몇 스텝마다 출력할지
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to add monitoring
        """
        # 기본 loss 계산
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        self.step_count += 1
        
        # 주기적으로만 출력
        if self.step_count % self.print_every_n_steps == 0:
            self._print_model_outputs(inputs, outputs, loss)
        
        return (loss, outputs) if return_outputs else loss
    

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        gen_kwargs = {
            "max_length": self.args.generation_max_length,
            "num_beams": self.args.generation_num_beams,
        }

        if self.restrict_decode_vocab is not None:
            processor = LogitsProcessorList(
                [RestrictVocabLogitsProcessor(self.restrict_decode_vocab)]
            )
            gen_kwargs["logits_processor"] = processor

        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )

    def _print_model_outputs(self, inputs, outputs, loss):
        """모델 출력을 깔끔하게 출력"""
        
        print("\n" + "="*80)
        
        # Loss 처리 - 텐서의 차원 확인
        if isinstance(loss, torch.Tensor):
            if loss.dim() == 0:  # 스칼라
                loss_value = loss.item()
            else:  # 벡터나 행렬
                loss_value = loss.mean().item()  # 평균값 사용
        else:
            loss_value = float(loss)
        
        print(f"[Step {self.step_count}] Loss: {loss_value:.4f}")
        print("="*80)
        
        # 1. 모델의 logits에서 예측 토큰 추출 (argmax)
        predictions = outputs.logits.argmax(dim=-1)
        
        # 2. 입력, 레이블, 예측 디코딩
        batch_size = min(3, predictions.shape[0])  # 최대 3개 샘플만 출력
        
        for i in range(batch_size):
            print(f"\n[Sample {i+1}]")
            
            # 입력 텍스트
            if "input_ids" in inputs:
                input_text = self.tokenizer.decode(
                    inputs["input_ids"][i], 
                    skip_special_tokens=False
                )
                print(f"📥 Input: {input_text[:100]}{'...' if len(input_text) > 100 else ''}")
            
            # 타겟 (레이블) 텍스트
            if "labels" in inputs:
                # -100을 pad_token_id로 교체
                labels = inputs["labels"][i]
                labels_clean = [
                    tok if tok != -100 else self.tokenizer.pad_token_id 
                    for tok in labels.tolist()
                ]
                target_text = self.tokenizer.decode(
                    labels_clean, 
                    skip_special_tokens=False
                )
                print(f"🎯 Target: {target_text[:100]}{'...' if len(target_text) > 100 else ''}")
            
            # 모델 예측 텍스트
            pred_text = self.tokenizer.decode(
                predictions[i], 
                skip_special_tokens=False
            )
            print(f"🤖 Predicted: {pred_text[:100]}{'...' if len(pred_text) > 100 else ''}")
            
            # 간단한 일치 여부 체크
            if 'target_text' in locals():
                if pred_text.strip() == target_text.strip():
                    print(f"✅ Exact match!")
                else:
                    # 처음 몇 글자가 일치하는지 확인
                    min_len = min(len(pred_text), len(target_text))
                    if min_len > 0:
                        match_chars = sum(1 for a, b in zip(pred_text[:min_len], target_text[:min_len]) if a == b)
                        match_ratio = match_chars / min_len
                        print(f"📊 Character match: {match_ratio:.1%} ({match_chars}/{min_len} chars)")
        
        print("="*80 + "\n")
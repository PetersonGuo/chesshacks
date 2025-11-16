# NNUE Model Training Fixes

## Problem
The NNUE model was not improving during training due to a critical output range mismatch.

## Root Cause

### Critical Issue: Output Activation Constraint
**Location:** `train/nnue_model/model.py:190-193` (old code)

The model used:
```python
x = torch.tanh(x) * self.output_scale  # output_scale = 3.0
```

This created an artificial limit on the output range to **[-3.0, 3.0]**.

**Why this broke training:**
1. Training data has evaluation scores from **-1500 to +1500 centipawns**
2. These get **z-score normalized** (mean ~0, std ~400-600)
3. After normalization, many values exceed [-3, 3]
4. **The model literally could not output correct values** for a large portion of the training data
5. Loss plateaued because the model hit its output ceiling/floor

### Secondary Issue: Conservative Weight Initialization
**Location:** `train/nnue_model/model.py:168`

Used `gain=0.5` in Xavier initialization, which made initial weights too small and slowed learning.

## Fixes Applied

### 1. Removed Output Activation Constraint ✅
**Changed:**
```python
# OLD (BROKEN)
x = torch.tanh(x) * self.output_scale
return x

# NEW (FIXED)
x = self.fc3(x)
# Linear output - allows full range of normalized evaluation scores
return x
```

**Impact:** Model can now output the full range of normalized values needed for training.

### 2. Improved Weight Initialization ✅
**Changed:**
```python
# OLD
nn.init.xavier_uniform_(m.weight, gain=0.5)

# NEW
nn.init.xavier_uniform_(m.weight, gain=1.0)
```

**Impact:** Better initial gradients for faster convergence.

### 3. Removed Unused Attribute ✅
Removed `self.output_scale = 3.0` as it's no longer needed.

## Expected Results

After these fixes, you should see:
- ✅ Loss actually decreasing during training
- ✅ Model able to fit the training data
- ✅ Validation loss improving over epochs
- ✅ Faster initial convergence

## Additional Recommendations

### Optional: Increase Model Capacity
Current architecture: `768 → 256 → 32 → 32 → 1`

The bottleneck from 256 → 32 is quite aggressive. Consider:
```python
# Current (smaller, faster)
hidden_size=256, hidden2_size=32, hidden3_size=32

# Recommended (better capacity)
hidden_size=256, hidden2_size=64, hidden3_size=32
```

You can modify this in `train/config.py`:
- For `RTX_5070_CONFIG`: Update lines 208-209
- For `RTX_5070_QUALITY_CONFIG`: Already uses 512→64→64 (good)

### Training Tips

1. **Start with a small test run** to verify the fix:
   ```bash
   python train/nnue_model/train.py --config fast
   ```

2. **Monitor the loss** - it should now decrease steadily

3. **If loss still plateaus**, try:
   - Reduce learning rate to 0.0003
   - Increase model size (hidden2_size=64)
   - Verify your data quality with the diagnostic script:
     ```bash
     python diagnose_nnue.py data/train.jsonl
     ```

4. **Check for overfitting**:
   - If train loss decreases but val loss increases: increase weight_decay
   - Current weight_decay is 1e-4 (reasonable)

## Files Modified

1. `train/nnue_model/model.py` - Fixed output activation and weight initialization
2. `diagnose_nnue.py` - Created diagnostic tool (NEW)
3. `NNUE_FIXES.md` - This documentation (NEW)

## Testing

To verify the fix works:

1. Run a quick test:
   ```bash
   python train/nnue_model/train.py --config fast
   ```

2. Watch for:
   - Loss decreasing (not flat)
   - Training loss < 0.5 after a few epochs
   - Validation loss improving

3. For full training:
   ```bash
   python train/nnue_model/train.py --config rtx5070
   ```

## Why This Fix Works

**Before:**
- Model output: `tanh(x) * 3` → bounded to [-3, 3]
- Normalized targets: range from -6 to +6 (or wider)
- Model can't match targets → loss stuck → no learning

**After:**
- Model output: linear (unbounded)
- Normalized targets: any range
- Model can match targets → loss decreases → learning happens

The loss function (MSE or Huber) provides the necessary regularization without needing to artificially bound the outputs.

## References

- Original NNUE paper uses linear outputs
- Stockfish NNUE uses clipped ReLU in hidden layers but linear output
- The tanh activation was likely a misunderstanding of output scaling

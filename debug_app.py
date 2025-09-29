#!/usr/bin/env python
"""
Debug script to test Week 3 module loading
"""

import os
import sys

# 프로젝트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.week02_cnn.cnn_module import CNNModule
    print("✅ Week 2 module imported successfully")
except Exception as e:
    print(f"❌ Week 2 import error: {e}")

try:
    from modules.week03.transfer_learning_module import TransferLearningModule
    print("✅ Week 3 module imported successfully")
except Exception as e:
    print(f"❌ Week 3 import error: {e}")

# Test module initialization
try:
    modules = {
        'Week 2: CNN': CNNModule(),
        'Week 3: Transfer Learning': TransferLearningModule(),
    }
    print(f"✅ Module dictionary created: {list(modules.keys())}")

    # Test if Week 3 module has required methods
    week3_module = modules['Week 3: Transfer Learning']
    print(f"✅ Week 3 module type: {type(week3_module)}")
    print(f"✅ Week 3 has render_ui method: {hasattr(week3_module, 'render_ui')}")
    print(f"✅ Week 3 has render method: {hasattr(week3_module, 'render')}")

except Exception as e:
    print(f"❌ Module initialization error: {e}")
    import traceback
    traceback.print_exc()

print("\nDebugging complete.")
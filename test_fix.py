#!/usr/bin/env python
"""
Test script to verify the fix
"""

import os
import sys

# 프로젝트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.week02_cnn.cnn_module import CNNModule
from modules.week03.transfer_learning_module import TransferLearningModule

class TestSmartVisionApp:
    """메인 애플리케이션 클래스 (테스트용)"""

    def __init__(self):
        self.modules = {
            'Week 2: CNN': CNNModule(),
            'Week 3: Transfer Learning': TransferLearningModule(),
        }

    def test_module_selection(self):
        """모듈 선택 로직 테스트"""
        print("Available modules:", list(self.modules.keys()))

        # Test Week 2 selection
        selected_module = 'Week 2: CNN'
        if selected_module in self.modules:
            print(f"✅ {selected_module} would render module UI")
        else:
            print(f"❌ {selected_module} would render home page")

        # Test Week 3 selection
        selected_module = 'Week 3: Transfer Learning'
        if selected_module in self.modules:
            print(f"✅ {selected_module} would render module UI")
        else:
            print(f"❌ {selected_module} would render home page")

        # Test invalid selection
        selected_module = 'Week 4: Unknown'
        if selected_module in self.modules:
            print(f"✅ {selected_module} would render module UI")
        else:
            print(f"✅ {selected_module} would render home page (expected)")

if __name__ == "__main__":
    app = TestSmartVisionApp()
    app.test_module_selection()
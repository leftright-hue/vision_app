#!/bin/bash
# Google Cloud 서비스 계정 설정 자동화 스크립트

echo "🔧 Google Cloud Video Intelligence API 설정 도우미"
echo "================================================="

# 1. Google Cloud Console URL 제공
echo ""
echo "📋 1단계: 서비스 계정 생성"
echo "다음 URL로 이동하여 서비스 계정을 생성하세요:"
echo "👉 https://console.cloud.google.com/iam-admin/serviceaccounts"
echo ""

echo "📝 서비스 계정 생성 정보:"
echo "  - 서비스 계정 이름: vision-app-service"
echo "  - 설명: Smart Vision App for Video Intelligence API"
echo "  - 역할: Cloud Video Intelligence API User"
echo ""

# 2. JSON 키 파일 확인
echo "🔍 2단계: JSON 키 파일 확인"
echo ""

# Downloads 폴더에서 JSON 키 파일 검색
DOWNLOADS_PATH="$HOME/Downloads"
JSON_FILES=$(find "$DOWNLOADS_PATH" -name "*.json" -mtime -1 2>/dev/null)

if [ ! -z "$JSON_FILES" ]; then
    echo "✅ 최근 다운로드된 JSON 파일들을 발견했습니다:"
    echo "$JSON_FILES"
    echo ""
    
    # 가장 최근 JSON 파일 선택
    LATEST_JSON=$(ls -t $JSON_FILES | head -n 1)
    echo "📁 가장 최근 파일: $LATEST_JSON"
    
    # 프로젝트 폴더로 복사
    PROJECT_DIR="/Users/callii/Documents/vision_app"
    TARGET_FILE="$PROJECT_DIR/google-cloud-key.json"
    
    read -p "이 파일을 Google Cloud 키로 사용하시겠습니까? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp "$LATEST_JSON" "$TARGET_FILE"
        chmod 600 "$TARGET_FILE"
        echo "✅ JSON 키 파일을 복사했습니다: $TARGET_FILE"
        
        # .env 파일 업데이트
        ENV_FILE="$PROJECT_DIR/.env"
        if grep -q "^GOOGLE_APPLICATION_CREDENTIALS=" "$ENV_FILE"; then
            sed -i '' "s|^GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=$TARGET_FILE|" "$ENV_FILE"
        else
            echo "GOOGLE_APPLICATION_CREDENTIALS=$TARGET_FILE" >> "$ENV_FILE"
        fi
        echo "✅ .env 파일을 업데이트했습니다."
        
        echo ""
        echo "🎉 설정 완료! Streamlit 앱을 재시작하세요."
    else
        echo "❌ 설정을 취소했습니다."
    fi
else
    echo "❌ Downloads 폴더에서 최근 JSON 파일을 찾을 수 없습니다."
    echo ""
    echo "📋 수동 설정 방법:"
    echo "1. Google Cloud Console에서 JSON 키를 다운로드하세요"
    echo "2. 다운로드한 파일을 다음 위치로 이동하세요:"
    echo "   /Users/callii/Documents/vision_app/google-cloud-key.json"
    echo "3. 이 스크립트를 다시 실행하세요"
fi

echo ""
echo "📚 상세 가이드:"
echo "https://cloud.google.com/video-intelligence/docs/common/auth"
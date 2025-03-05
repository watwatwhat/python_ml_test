from datetime import datetime

# 現在の日時を取得
now = datetime.now()

# フォーマットして表示
print("現在の日時:", now.strftime("%Y-%m-%d %H:%M:%S"))
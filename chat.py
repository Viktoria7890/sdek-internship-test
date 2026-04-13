import urllib.request
import urllib.error
import json
import uuid
import sys

BASE_URL = "http://localhost:8000"
session_id = str(uuid.uuid4())[:8]


def send_message(message: str) -> str:
    data = json.dumps({"session_id": session_id, "message": message}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["response"]


def main():
    print("=" * 50)
    print("  CdekStart — консультант по стажировке")
    print("=" * 50)
    print(f"  Сессия: {session_id}")
    print("  Введите 'выход' или Ctrl+C для завершения")
    print("=" * 50)
    print()

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nДо свидания!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("выход", "exit", "quit"):
            print("До свидания!")
            sys.exit(0)

        try:
            response = send_message(user_input)
            print(f"\nБот: {response}\n")
        except Exception as e:
            print(f"\n[Ошибка] Не удалось получить ответ: {e}\n")


if __name__ == "__main__":
    main()

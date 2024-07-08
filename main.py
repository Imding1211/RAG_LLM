from database_controller import populate_database, clear_database
from query_controller import query_rag
import argparse

def run():
    while True:
        query_text = input("Enter your question or enter exit to stop:\n")

        if query_text == "exit":
            break

        query_rag(query_text)
        print("\n")

def populate(reset):
    if reset:
        clear_database()

    populate_database()
        

def main():
    parser = argparse.ArgumentParser(description="程式描述")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 定義 run 子命令
    parser_run = subparsers.add_parser('run', help='執行程式')
    parser_run.set_defaults(func=lambda args: run())

    # 定義 populate 子命令
    parser_populate = subparsers.add_parser('populate', help='更新資料庫')
    parser_populate.add_argument('--reset', action='store_true', help='重置資料庫')
    parser_populate.set_defaults(func=lambda args: populate(args.reset))

    args = parser.parse_args()

    # 根據子命令呼叫對應的函式
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

import os
import time

from Train import Tran


def method_name(start_time, end_time, fold):
    total_training_time_seconds = end_time - start_time

    total_training_time_hours = total_training_time_seconds / 3600

    time_file_path = f'train_time/{eventlog}/first_training_time_{fold}.txt'
    os.makedirs(f'train_time/{eventlog}', exist_ok=True)

    with open(time_file_path, 'w') as time_file:
        time_file.write(f"training time: {total_training_time_hours:.3f} hours\n")

    print("-" * 90)
    print("\n")

    print(f"{fold} fold---Total training time: {total_training_time_hours:.3f} hours")


if __name__ == "__main__":
    list_eventlog = [
        'bpi13_closed_problems',
        'bpi13_problems',
        'bpi13_incidents',
        'bpi12w_complete',
        'bpi12_all_complete',
        'BPI2020_Prepaid',
    ]

    for eventlog in list_eventlog:

        print(f"--------------开始-记录时间------------")

        start_total = time.perf_counter()

        print(f"--------------预训练模型: {eventlog} ------------")
        for i in range(4):
            Tran(eventlog, choice=i).tran_main()

        end_total = time.perf_counter()
        method_name(start_total, end_total, 123)
        print(f"--------------结束-记录时间------------")

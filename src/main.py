from data_sampler import PairDataSampler
import faulthandler
import tracemalloc


def sampling(user1_name, user2_name, device1_address, device2_address):

    sampler = PairDataSampler(user1_name, user2_name, device1_address, device2_address)
    sampler.run()
    device1_data, device2_data = sampler.get_data()
    device1_data.to_csv(f"{user1_name}.csv")
    device2_data.to_csv(f"{user2_name}.csv")


def main():
    user1_name = "hoge"
    user2_name = "huga"
    device1_address = "A59901F2-0211-6282-AA8C-4858238B4AE0"
    device2_address = "F76B7A81-43CD-5515-B7C3-997B6847F307"

    tracemalloc.start()
    faulthandler.enable()

    sampling(user1_name, user2_name, device1_address, device2_address)

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("traceback")

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
        for line in stat.traceback.format():
            print(line)
        print("=====")


if __name__ == "__main__":
    main()

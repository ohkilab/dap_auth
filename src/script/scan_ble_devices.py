import asyncio
from bleak import BleakScanner


async def run():
    devices = await BleakScanner.discover()
    find = []
    try:
        for d in devices:
            # HC-06 / HC-04
            if d.name is not None and "HC-" in d.name:
                find.append(d)
                print(d)
        if len(find) == 0:
            print("No devices found in this search!")
    except Exception as ex:
        print("Bluetooth search failed to start")
        print(ex)


loop = asyncio.get_event_loop()
loop.run_until_complete(run())

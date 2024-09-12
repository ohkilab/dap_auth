import asyncio
from time import sleep
from bleak import BleakClient

# address = "F76B7A81-43CD-5515-B7C3-997B6847F307"
address = "A59901F2-0211-6282-AA8C-4858238B4AE0"
uuid = "49535343-8841-43f4-a8d4-ecbe34729bb3"


def get_writeBytes(regAddr, rValue):
    tempBytes = [None] * 5
    tempBytes[0] = 0xFF
    tempBytes[1] = 0xAA
    tempBytes[2] = regAddr
    tempBytes[3] = rValue & 0xFF
    tempBytes[4] = rValue >> 8
    return tempBytes


async def writeData(client, uuid, data):
    unlock_cmd = get_writeBytes(0x69, 0xB588)
    save_cmd = get_writeBytes(0x00, 0x0000)

    await sendData(client, uuid, unlock_cmd)
    sleep(0.1)
    await sendData(client, uuid, data)
    sleep(0.1)
    await sendData(client, uuid, save_cmd)
    sleep(0.1)


async def sendData(client, uuid, data):
    try:
        await client.write_gatt_char(uuid, bytes(data))
    except Exception as ex:
        print(ex)


async def run(address, uuid, loop):
    async with BleakClient(address, loop=loop) as client:

        x = client.is_connected
        print("Connected: {0}".format(x))

        turn_on_cmd = get_writeBytes(0x1B, 0x0001)
        turn_off_cmd = get_writeBytes(0x1B, 0x0000)

        print("on")
        await writeData(client, uuid, turn_on_cmd)
        sleep(1)
        print("off")
        await writeData(client, uuid, turn_off_cmd)
        sleep(1)


loop = asyncio.get_event_loop()
loop.run_until_complete(run(address, uuid, loop))

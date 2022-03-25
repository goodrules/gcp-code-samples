/**
 * A transforms incoming device signal event to BQ schema event
 * @param {string} inJson
 * @return {string} outJson
 **/

function transformDeviceSignalEvent(inJson) {
  var original = JSON.parse(inJson)

  var transformed = {
    eventId: original.eventId,
    deviceId: original.deviceId,
    eventTime: original.eventTime,
    city: original.city,
    temp: original.temp,
    flowrate: original.flowrate
  }

  return JSON.stringify(transformed)
}
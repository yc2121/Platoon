#!/usr/bin/python
# -*- coding: UTF-8 -*-

class Vehicle:
    # Vehicle simulator: include all the information for a vehicle

    def __init__(self, id, startPosition, startDirection, velocity, platoonId, vehicleType):
        self.Id = id
        self.Position = startPosition
        self.Direction = startDirection
        self.Velocity = velocity
        self.PlatoonId = platoonId
        self.VehicleType = vehicleType 
from django.db import models

# class InputData(models.Model):
#     temperature = models.FloatField()
#     humidity = models.FloatField()
#     tvoc = models.FloatField()
#     eco2 = models.FloatField()
#     raw_h2 = models.FloatField()
#     raw_ethanol = models.FloatField()
#     pressure = models.FloatField()
#     pm1 = models.FloatField()
#     pm2_5 = models.FloatField()
#     nc0_5 = models.FloatField()
#     nc1_0 = models.FloatField()
#     nc2_5 = models.FloatField()
#     cnt = models.FloatField()

#     def __str__(self):
#         return f'Temperature: {self.temperature}, Humidity: {self.humidity}'





from django.db import models

class InputData(models.Model):
    Age = models.IntegerField()
    Flight_Distance = models.FloatField()
    Inflight_wifi_service = models.IntegerField()
    Departure_Arrival_time_convenient = models.IntegerField()
    Ease_of_Online_booking = models.IntegerField()
    Gate_location = models.IntegerField()
    Food_and_drink = models.IntegerField()
    Online_boarding = models.IntegerField()
    Seat_comfort = models.IntegerField()
    Inflight_entertainment = models.IntegerField()
    On_board_service = models.IntegerField()
    Leg_room_service = models.IntegerField()
    Baggage_handling = models.IntegerField()
    Checkin_service = models.IntegerField()
    Inflight_service = models.IntegerField()
    Cleanliness = models.IntegerField()
    Departure_Delay_in_Minutes = models.FloatField()
    Arrival_Delay_in_Minutes = models.FloatField()
    Gender_Female = models.BooleanField()
    Gender_Male = models.BooleanField()
    Type_of_Travel_Business_travel = models.BooleanField()
    Type_of_Travel_Personal_Travel = models.BooleanField()
    Class_Business = models.BooleanField()
    Class_Eco = models.BooleanField()
    Class_Eco_Plus = models.BooleanField()

    def __str__(self):
        return f'Age: {self.Age}, Flight Distance: {self.Flight_Distance}'

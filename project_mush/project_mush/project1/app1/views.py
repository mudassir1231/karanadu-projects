
# Create your views here.
from django.shortcuts import render ,HttpResponse
from .forms import InputDataForm
from .models import InputData
import os
# from sklearn.externals import joblib
import joblib
# Load the trained SVM classifier
    # Define the path to your .pkl file
pkl_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'rf_classifier_model.pkl')

svm_classifier = joblib.load(pkl_file_path)

# Load the scaler used during training
scale_pkl_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'scaler.pkl')

scaler = joblib.load(scale_pkl_file_path)

def predict_fire_alarm(request):
    if request.method == 'POST':
        form = InputDataForm(request.POST)
        if form.is_valid():
            input_data = form.save(commit=False)

            # Standardize the input data using the loaded scaler
            # input_data_scaled = scaler.transform([[input_data.temperature, input_data.humidity, input_data.tvoc, input_data.eco2, input_data.raw_h2, input_data.raw_ethanol, input_data.pressure, input_data.pm1, input_data.pm2_5, input_data.nc0_5, input_data.nc1_0, input_data.nc2_5, input_data.cnt]])
            input_data_scaled = scaler.transform([[input_data.Age, input_data.Flight_Distance, input_data.Inflight_wifi_service, input_data.Departure_Arrival_time_convenient, input_data.Ease_of_Online_booking, input_data.Gate_location, input_data.Food_and_drink, input_data.Online_boarding, input_data.Seat_comfort, input_data.Inflight_entertainment, input_data.On_board_service, input_data.Leg_room_service, input_data.Baggage_handling, input_data.Checkin_service, input_data.Inflight_service, input_data.Cleanliness, input_data.Departure_Delay_in_Minutes, input_data.Arrival_Delay_in_Minutes, input_data.Gender_Female, input_data.Gender_Male, input_data.Type_of_Travel_Business_travel, input_data.Type_of_Travel_Personal_Travel, input_data.Class_Business, input_data.Class_Eco, input_data.Class_Eco_Plus]])

            print("data is :",input_data_scaled)
            # Make predictions
            prediction = svm_classifier.predict(input_data_scaled)
            # input_data.fire_alarm = prediction[0]

            # input_data.save()

            return render(request, 'result.html', {'input_data': prediction[0]})
    else:
        form = InputDataForm()
    
    return render(request, 'predict_form.html', {'form': form})







def home(request):
    return HttpResponse("hello")



def home1(request):
    return render(request,'result.html')

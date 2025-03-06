#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>  // For std::max

class PIDController {
public:
    PIDController(double Kp, double Ki, double Kd, double target) 
        : Kp(Kp), Ki(Ki), Kd(Kd), target(target), previousError(0), integral(0) {}

    // Update PID controller and calculate the control output
    double update(double currentValue) {
        // Calculate the current error (target value - current value)
        double error = target - currentValue;

        // Integrate the error for the integral term
        integral += error;

        // Calculate the derivative (rate of change of error)
        double derivative = error - previousError;

        // Calculate the PID control output
        double output = Kp * error + Ki * integral + Kd * derivative;

        // Clamp the output to be >= 0 (because heater power can't be negative)
        output = std::max(output, 0.0);

        // Save the current error for the next calculation (for derivative term)
        previousError = error;

        return output;
    }

private:
    double Kp, Ki, Kd;  // PID parameters
    double target;       // Target value (desired temperature)
    double previousError; // Previous error for derivative calculation
    double integral;      // Integral term
};

int main(int argc, char * argv[]) {
    // Target temperature is 60°C
    double targetTemperature = 60.0;
    double currentTemperature = 55.0;  // Initial temperature

    double lossrate = 0.02;
    if(argc == 2) {
        lossrate = std::stoi(argv[1])/100.0;
    }
    // PID parameters (these values need to be adjusted for real system)
    double Kp = 2.0;  // Proportional gain
    double Ki = 0.1;  // Integral gain
    double Kd = 1.0;  // Derivative gain

    // Create a PID controller instance
    PIDController pid(Kp, Ki, Kd, targetTemperature);
    int i = 0;
    // Simulate the temperature control process over 100 time steps
    while (abs( targetTemperature - currentTemperature ) > 1e-5) {
        // Get the PID control output (e.g., adjust the heater power)
        double controlSignal = pid.update(currentTemperature);
	std::cout << "control signal= " << controlSignal << std::endl;

        // Apply the control signal to adjust the current temperature
        // Assume the control signal increases the temperature by 'controlSignal * 0.1' (simplified)
        currentTemperature += controlSignal * 0.1; 

        // Account for the natural temperature loss (1°C per minute)
        currentTemperature *= 1 - lossrate;

        // Output the current temperature
        std::cout << "Step " << i++ + 1 << ": Current Temperature = " << currentTemperature << "°C" << std::endl;

        // Simulate time delay (e.g., 1 minute)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Just to simulate time passing
    }

    return 0;
}


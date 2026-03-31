class VehicleController:
    def control(self, decision):
        action = decision.get("action", "Proceed")
        
        commands = {
            "throttle": 0.0,
            "brake": 0.0,
            "steering": 0.0
        }
        
        if action == "Brake":
            commands["throttle"] = 0.0
            commands["brake"] = 0.8
        elif action == "Slow":
            commands["throttle"] = 0.2
            commands["brake"] = 0.0
        elif action == "TurnLeft":
            commands["throttle"] = 0.3
            commands["steering"] = -0.3
        elif action == "TurnRight":
            commands["throttle"] = 0.3
            commands["steering"] = 0.3
        elif action == "Proceed":
            commands["throttle"] = 0.5
            commands["brake"] = 0.0
            
        return commands

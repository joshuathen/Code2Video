from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "The Cheetah's Journey: Speed vs. Distance"
        lecture_lines = [
            "Meet our cheetah racing across the savanna.",
            "Speedometer shows its instantaneous speed, the derivative.",
            "Odometer tracks total distance covered, the integral."
        ]
        
        # Setup layout
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        CHEETAH_COLOR = "#D4AF37"
        SPEEDOMETER_COLOR = "#FFFFFF"
        ODOMETER_COLOR = "#FFFFFF"
        DERIVATIVE_COLOR = "#00FFFF"
        INTEGRAL_COLOR = "#FFD700"
        
        # ValueTrackers for synchronization
        time_tracker = ValueTracker(0)
        
        # Functions to drive animations
        def get_speed():
            t = time_tracker.get_value()
            return 1.5 + 0.5 * np.sin(2 * t)
            
        def get_distance():
            t = time_tracker.get_value()
            # Integral of 1.5 + 0.5*sin(2t) is 1.5t - 0.25*cos(2t) + C
            # Scale by 0.5 to keep it within the grid width
            return (1.5 * t - 0.25 * np.cos(2 * t) + 0.25) * 0.5

        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.lecture[0].set_color(CHEETAH_COLOR)
        
        # Cheetah silhouette [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg]
        # Using a fallback rectangle if loading fails, but according to MAS instructions, assume it exists.
        try:
            cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
            cheetah.set_color(CHEETAH_COLOR)
        except:
            cheetah = Triangle(color=CHEETAH_COLOR, fill_opacity=1).scale(0.2).rotate(-PI/2)
        
        cheetah.scale(0.4)
        cheetah.move_to(self.grid["B1"])
        
        # Movement updater
        cheetah.add_updater(lambda m: m.move_to(self.grid["B1"] + RIGHT * get_distance()))
        
        self.play(FadeIn(cheetah))
        self.play(time_tracker.animate.set_value(2), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Update lecture colors
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(DERIVATIVE_COLOR)
        
        # Speedometer: Circle with a needle
        speedometer_base = Circle(radius=0.4, color=SPEEDOMETER_COLOR)
        # Fix: Issue 27 - Move to C2
        self.place_at_grid(speedometer_base, "C2")
        
        # Needle
        needle = Line(
            speedometer_base.get_center(), 
            speedometer_base.get_center() + UP * 0.35, 
            color=RED, 
            stroke_width=4
        )
        # Angle updater based on speed
        needle.add_updater(lambda m: m.set_angle(PI/2 - (get_speed() - 1.5) * PI/2))
        
        speed_label = Text("Speed (Derivative)", font_size=22, color=DERIVATIVE_COLOR)
        # Fix: Issue 27 - Move to C3, Issue 28 - scale_factor=0.6
        self.place_at_grid(speed_label, "C3", scale_factor=0.6)
        
        self.play(Create(speedometer_base), Create(needle), Write(speed_label))
        self.play(time_tracker.animate.set_value(4), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Update lecture colors
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(INTEGRAL_COLOR)
        
        # Odometer: Box with increasing digits
        odometer_box = Rectangle(width=1.2, height=0.4, color=ODOMETER_COLOR)
        # Fix: Issue 26 - Move to E2
        self.place_at_grid(odometer_box, "E2")
        
        distance_val = DecimalNumber(0, num_decimal_places=2, color=WHITE, font_size=22)
        distance_val.add_updater(lambda m: m.set_value(get_distance()))
        distance_val.move_to(odometer_box.get_center())
        
        odo_label = Text("Distance (Integral)", font_size=22, color=INTEGRAL_COLOR)
        # Fix: Issue 26 - Move to E3, Issue 28 - scale_factor=0.6
        self.place_at_grid(odo_label, "E3", scale_factor=0.6)
        
        self.play(Create(odometer_box), Write(distance_val), Write(odo_label))
        self.play(time_tracker.animate.set_value(6), run_time=3, rate_func=linear)
        self.wait(1)

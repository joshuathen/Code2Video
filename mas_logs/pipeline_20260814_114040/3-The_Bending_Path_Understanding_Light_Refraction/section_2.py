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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Nature of Light Speed", 
                          ["Light speed changes in different materials.", 
                           "Air is fast, water is slower.", 
                           "This speed change causes bending."])
        
        # Load Assets
        glass_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")
        speedometer_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        
        # Setup visual elements
        self.place_at_grid(glass_asset, 'C4', scale_factor=0.5)
        
        # Pulse
        pulse = Circle(radius=0.3, color="#00FFFF").set_fill(color="#00FFFF", opacity=0.8)
        self.place_in_area(pulse, 'A4', 'C6', scale_factor=0.6)
        
        # Speedometer
        speed_text = Text("Speed:", font_size=20)
        speed_val = DecimalNumber(100, num_decimal_places=0, color="#00FFFF")
        speedometer_group = VGroup(speedometer_asset, speed_text, speed_val).arrange(DOWN)
        self.place_at_grid(speedometer_group, 'D5', scale_factor=0.7)
        
        # Label for speed value
        speedometer_label = Text("Speed Value:", font_size=18, color="#00FFFF")
        self.place_at_grid(speedometer_label, 'D4', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(FadeIn(pulse), FadeIn(glass_asset), FadeIn(speedometer_group), Write(speedometer_label))
        # Move pulse to boundary (glass)
        self.play(pulse.animate.move_to(self.grid['B4']), run_time=3)
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        # Pulse enters slower medium, change speed
        speed_val.add_updater(lambda d: d.set_value(60))
        self.play(
            pulse.animate.move_to(self.grid['C4']), 
            run_time=4, 
            rate_func=linear
        )
        speed_val.remove_updater(lambda d: d.set_value(60))
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        # Bending indication
        bend_arrow = Arrow(start=self.grid['C4'], end=self.grid['E6'], color="#FFFF00")
        self.play(Create(bend_arrow))
        self.wait(4)

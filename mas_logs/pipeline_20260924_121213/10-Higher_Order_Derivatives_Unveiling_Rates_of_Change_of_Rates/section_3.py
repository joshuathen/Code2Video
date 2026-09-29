from manim import *
import os

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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Physical Interpretation: Acceleration", [
            "In physics, first derivative is velocity.", 
            "Second derivative represents acceleration.", 
            "G-force relates to the second derivative."
        ])
        
        # Elements
        car_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        
        velocity_label = Text("v(t) = s'(t)", font_size=24, color="#00FFFF")
        accel_label = Text("a(t) = v'(t) = s''(t)", font_size=24, color="#FFFF00")
        
        # === Animation for Lecture Line 1 ===
        # Show a car moving with increasing velocity
        self.lecture[0].set_color("#FFFFFF")
        car = self.place_at_grid(car_icon.copy(), "B4", scale_factor=0.5)
        self.play(FadeIn(car))
        self.play(car.animate.shift(RIGHT * 1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Plot position s(t) and its tangent slope + Label slope as velocity
        self.lecture[1].set_color("#FF00FF")
        self.play(Write(self.place_in_area(velocity_label, "D3", "D5", scale_factor=0.7)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Label acceleration + Display summary with car icon
        self.lecture[2].set_color("#808080")
        self.play(Write(self.place_in_area(accel_label, "E3", "E5", scale_factor=0.7)))
        car_summary = self.place_at_grid(car_icon.copy(), "F5", scale_factor=0.3)
        self.play(FadeIn(car_summary))
        self.wait(2)

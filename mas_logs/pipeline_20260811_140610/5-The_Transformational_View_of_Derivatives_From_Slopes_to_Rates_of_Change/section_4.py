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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: The Predictive Power of Rates", 
                          ["Derivatives allow predicting future system states.", 
                           "We determine the next trajectory path.", 
                           "Drones adjust rotors using this data."])
        
        # Load Assets
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg")
        
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, "A2", "B6", scale_factor=0.85)
        
        curve = axes.plot(lambda x: 0.2 * x**2, color="#E67E22")
        
        # Drone moving
        drone_group = VGroup(drone).scale(0.2)
        drone_group.move_to(axes.c2p(0, 0))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#E67E22")
        self.play(Create(axes), Create(curve), FadeIn(drone_group), run_time=2)
        self.play(MoveAlongPath(drone_group, curve), run_time=3)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#3498DB")
        dot = Dot(color="#3498DB")
        dot.move_to(axes.c2p(3, 0.2 * 3**2))
        
        slope_line = Line(start=ORIGIN, end=RIGHT*1.5, color="#3498DB")
        slope_line.rotate(np.arctan(0.2 * 2 * 3))
        slope_line.move_to(dot.get_center())
        
        self.play(FadeIn(dot), Create(slope_line), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#2ECC71")
        velocity_axes = Axes(x_range=[0, 5, 1], y_range=[0, 2, 0.5], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(velocity_axes, "D2", "E6", scale_factor=0.85)
        
        velocity_curve = velocity_axes.plot(lambda x: 0.4 * x, color="#2ECC71")
        
        self.play(Create(velocity_axes), Create(velocity_curve), run_time=2)
        self.play(drone_group.animate.move_to(velocity_axes.c2p(3, 0.4 * 3)), run_time=2)

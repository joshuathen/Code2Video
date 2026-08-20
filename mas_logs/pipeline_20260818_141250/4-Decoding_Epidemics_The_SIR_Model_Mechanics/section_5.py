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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Simulation & Application: Flattening the Curve", 
                          ["Parameters change the outbreak shape.", 
                           "Lowering contact flattens the infection curve.", 
                           "Models help manage health system capacity."])
        
        # Create graphs for simulation
        axes = Axes(x_range=[0, 10, 1], y_range=[0, 1, 0.2], axis_config={"include_tip": False}).scale(0.5)
        self.place_in_area(axes, "B1", "E6")
        
        # Spiky curve
        spiky_curve = axes.plot(lambda x: 0.8 * np.exp(-(x-3)**2 / 0.5), color=RED)
        # Flat curve
        flat_curve = axes.plot(lambda x: 0.4 * np.exp(-(x-4)**2 / 2.0), color=GREEN)
        
        label_spiky = Text("No Intervention", color=RED, font_size=18)
        label_flat = Text("Social Distancing", color=GREEN, font_size=18)
        
        self.place_at_grid(label_spiky, "A2")
        self.place_at_grid(label_flat, "F2")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(axes), Write(label_spiky))
        self.play(Create(spiky_curve))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.play(Write(label_flat))
        self.play(Create(flat_curve))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        capacity_line = axes.plot(lambda x: 0.3, color=BLUE)
        label_cap = Text("Health Capacity", color=BLUE, font_size=18)
        self.place_at_grid(label_cap, "A5")
        self.play(Create(capacity_line), Write(label_cap))
        self.wait(2)

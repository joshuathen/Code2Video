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

class Section6Scene(TeachingScene):
    def construct(self):
        # Define lecture lines based on storyboard
        lecture_lines = [
            "- Large models contain thousands of these Key-Value pairs.",
            "- Knowledge is distributed across many layers and neurons.",
            "- This massive web stores the vast information GPT models hold."
        ]
        
        # Setup the scene layout
        self.setup_layout("Scaling Up: Billions of Memories", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Color the first line to match the first visual set
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        
        # Start with a single "neuron" (key-value pair representation)
        single_neuron = Circle(radius=0.3, color="#ADD8E6", fill_opacity=1)
        # Fix for Issue 42: Precise positioning using grid-point anchoring
        self.place_at_grid(single_neuron, 'C3', scale_factor=1.2)
        self.play(Create(single_neuron))
        self.wait(0.5)
        
        # Create a 10x10 grid of neurons
        neurons_10x10 = VGroup(*[
            Circle(radius=0.1, color="#ADD8E6", fill_opacity=1, stroke_width=0)
            for _ in range(100)
        ]).arrange_in_grid(rows=10, cols=10, buff=0.1)
        
        # Fix for Issue 41: occupy less vertical extent to avoid crowding
        self.place_in_area(neurons_10x10, "B1", "E6", scale_factor=0.85)
        
        # Transition: Transform single neuron into the 10x10 grid (zoom out effect)
        self.play(ReplacementTransform(single_neuron, neurons_10x10), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color the second line to match the second visual set
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Create a massive dense point cloud (point cloud representation)
        # Using Dot is more efficient for large quantities
        point_cloud = VGroup(*[
            Dot(radius=0.03, color="#00FFFF")
            for _ in range(1024) # 32x32 grid
        ]).arrange_in_grid(rows=32, cols=32, buff=0.04)
        
        # Fix for Issue 41: occupy less vertical extent to avoid crowding
        self.place_in_area(point_cloud, 'B1', 'E6', scale_factor=0.85)
        
        # Rapidly zoom out further: transform the 10x10 grid into the dense point cloud
        self.play(
            ReplacementTransform(neurons_10x10, point_cloud),
            run_time=2
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Color the third line to match the final overlay
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # Overlay the "175 Billion Parameters" text
        param_text = Text("175 Billion Parameters", color="#FFD700", font_size=32)
        
        # Background rectangle for readability
        bg_rect = Rectangle(
            width=param_text.width + 0.6,
            height=param_text.height + 0.6,
            color=BLACK,
            fill_opacity=0.8,
            stroke_width=0
        )
        text_group = VGroup(bg_rect, param_text)
        
        # Fix for Issue 40: Adjust scale and position to avoid overlap and obstruction
        self.place_in_area(text_group, 'E1', 'F6', scale_factor=0.75)
        
        # Animate the text appearing over the point cloud
        self.play(FadeIn(text_group))
        self.wait(4)

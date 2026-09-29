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
        self.setup_layout("Properties and Constraints", [
            "PDFs must be non-negative everywhere.",
            "Total integral over the domain must equal one.",
            "Normalization ensures total probability equals unity."
        ])
        
        # Setup Axes
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 2, 0.5],
            x_length=4,
            y_length=3,
            axis_config={"include_tip": True}
        )
        
        # Applying requested improvements from issues 29, 30, 31, 38
        self.place_at_grid(axes, 'C5', scale_factor=0.9)
        self.add(axes)

        # Graph: f(x) > 1 initially (e.g., peak at 1.5)
        graph = axes.plot(lambda x: 1.5 * np.exp(-(x-2)**2), color=BLUE)
        
        # Create a container for graph elements for easier placement
        area = axes.get_area(graph, [0, 5], color=YELLOW, opacity=0.4)
        graph_group = VGroup(axes, graph, area)
        self.place_in_area(graph_group, 'B3', 'E5', scale_factor=0.85)

        # Labels
        label_total = Text("Total Area > 1", font_size=20)
        self.place_at_grid(label_total, 'A5', scale_factor=0.7)
        self.add(label_total)
        
        # === Animation for Lecture Line 1 ===
        # Emphasize f(x) >= 0 in #FF0000
        self.play(self.lecture[0].animate.set_color('#FF0000'))
        glow = graph.copy().set_stroke(width=6, color=RED, opacity=0.5)
        self.play(ShowPassingFlash(glow), run_time=1.5)

        # === Animation for Lecture Line 2 ===
        # Show total area > 1, then normalize to 1
        self.play(self.lecture[1].animate.set_color('#00FF00'))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Animate normalization
        self.play(self.lecture[2].animate.set_color('#00FFFF'))
        
        # Normalize curve
        new_graph = axes.plot(lambda x: 0.9 * np.exp(-(x-2)**2), color=BLUE)
        new_area = axes.get_area(new_graph, [0, 5], color=YELLOW, opacity=0.4)
        
        new_label = Text("Total Probability = 1", font_size=20, color=WHITE)
        self.place_at_grid(new_label, 'A5', scale_factor=0.7)
        
        self.play(
            Transform(graph, new_graph),
            Transform(area, new_area),
            Transform(label_total, new_label),
            run_time=2
        )
        self.wait(2)

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
        # 1. Setup layout with Section Title and Lecture Lines
        title_text = "Prerequisite: The Intuitive Idea of a Limit"
        lecture_lines = [
            "Limits describe behavior as we approach a point.",
            "Even if a point is missing, the trend remains.",
            "Imagine a squirrel approaching a hidden nut.",
            "From both sides, the path points to height L.",
            "The limit depends on the path, not the destination."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # "Limits describe behavior as we approach a point."
        # Visualization: Axes and a linear function graph.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        axes = Axes(
            x_range=[0, 8, 1],
            y_range=[0, 5, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, 'A1', 'F6', scale_factor=1.0)
        
        # Simple linear function: f(x) = 0.3x + 1.5. At x=5, f(5)=3.0
        def func(x):
            return 0.3 * x + 1.5
            
        full_graph = axes.plot(func, x_range=[0.5, 7.5], color=WHITE)
        
        self.play(
            Write(axes),
            Create(full_graph)
        )

        # === Animation for Lecture Line 2 ===
        # "Even if a point is missing, the trend remains."
        # Visualization: Replace the full graph with a graph having a hole at x=5.
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        hole_pos = axes.c2p(5, 3.0)
        # Using a Dot with no fill to represent a hole (discontinuity)
        hole = Dot(hole_pos, radius=0.08, color=WHITE, fill_opacity=0, stroke_width=2)
        
        # Split graph into two parts to emphasize the gap
        graph_left = axes.plot(func, x_range=[0.5, 4.85], color=WHITE)
        graph_right = axes.plot(func, x_range=[5.15, 7.5], color=WHITE)
        
        hole_label = Text("Point of Interest", font_size=18, color=WHITE)
        self.place_at_grid(hole_label, 'B5', scale_factor=1.0)
        
        self.play(
            FadeOut(full_graph),
            Create(graph_left), 
            Create(graph_right), 
            FadeIn(hole),
            Write(hole_label)
        )

        # === Animation for Lecture Line 3 ===
        # "Imagine a squirrel approaching a hidden nut."
        # Visualization: Two orange dots representing the squirrels on the path.
        self.play(self.lecture[2].animate.set_color("#FFA500"))
        
        dot_left = Dot(axes.c2p(1, func(1)), color="#FFA500")
        dot_right = Dot(axes.c2p(7, func(7)), color="#FFA500")
        
        self.play(FadeIn(dot_left), FadeIn(dot_right))

        # === Animation for Lecture Line 4 ===
        # "From both sides, the path points to height L."
        # Visualization: Animate dots toward the hole and show the limit height L.
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        
        # Target positions near the hole
        target_left = axes.c2p(4.7, func(4.7))
        target_right = axes.c2p(5.3, func(5.3))
        
        dashed_line = DashedLine(
            start=hole_pos, 
            end=axes.c2p(0, 3.0), 
            color="#00FFFF"
        )
        limit_label = Text("L = 3", color="#00FFFF", font_size=32)
        self.place_at_grid(limit_label, 'C1', scale_factor=1.0)
        
        self.play(
            dot_left.animate.move_to(target_left),
            dot_right.animate.move_to(target_right),
            Create(dashed_line),
            Write(limit_label),
            run_time=2
        )

        # === Animation for Lecture Line 5 ===
        # "The limit depends on the path, not the destination."
        # Final highlight and pause.
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(3)

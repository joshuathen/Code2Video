from manim import *
import numpy as np

# Define CYAN as it is not a default color in the Manim global namespace
CYAN = "#00FFFF"

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
        title_text = "Shift in Perspective: From Slope to Scaling"
        lecture_lines = [
            "We usually view derivatives as slopes on a graph.",
            "But derivatives also describe how functions stretch space.",
            "This is known as the transformational view of derivatives."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create a 2D coordinate system
        axes = Axes(
            x_range=[-2, 2],
            y_range=[-1, 3],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False, "color": WHITE}
        )
        curve = axes.plot(lambda x: 0.5 * x**2 + 0.5, x_range=[-2, 2], color=WHITE)
        
        # Tangent line at x=1
        dot = Dot(axes.c2p(1, 1), color=YELLOW)
        tangent = TangentLine(curve, alpha=0.75, length=3, color=YELLOW) 
        
        graph_group = VGroup(axes, curve, dot, tangent)
        self.place_in_area(graph_group, "B2", "E5", scale_factor=0.6)
        
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(dot), Create(tangent))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(RED)
        
        # Morph 2D graph into a single horizontal white number line
        number_line = NumberLine(
            x_range=[-3, 3],
            length=5,
            color=WHITE,
            include_numbers=False,
            include_tip=True
        )
        self.place_in_area(number_line, "D3", "D4", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(graph_group, number_line),
            run_time=1.5
        )
        
        # Introduce a red box labeled 'Stretching Machine'
        machine_rect = Rectangle(width=2, height=1, color=RED, fill_opacity=0.2)
        machine_label = Text("Stretching Machine", font_size=18, color=RED)
        machine = VGroup(machine_rect, machine_label)
        self.place_in_area(machine, "C3", "C4", scale_factor=1.0)
        
        self.play(FadeIn(machine))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(CYAN)
        
        # Animate a cyan segment passing through the machine
        segment = Line(LEFT * 0.2, RIGHT * 0.2, color=CYAN, stroke_width=6)
        self.place_at_grid(segment, "D1")
        
        self.play(FadeIn(segment))
        
        # Sliding through the machine
        self.play(segment.animate.move_to(self.grid["D2"]), run_time=0.5)
        
        # Stretched segment at the output
        stretched_segment = Line(LEFT * 0.4, RIGHT * 0.4, color=CYAN, stroke_width=6)
        self.place_at_grid(stretched_segment, "D5")
        
        # Transition: move to D3 (center), then transform to stretched at D5
        self.play(
            segment.animate.move_to(self.grid["D3"]),
            run_time=0.3
        )
        self.play(
            ReplacementTransform(segment, stretched_segment),
            run_time=0.7
        )
        
        # Highlight a specific point in yellow and show 'Scale: 2x'
        highlight_dot = Dot(color=YELLOW)
        self.place_in_area(highlight_dot, "D3", "D4", scale_factor=1.0)
        
        scale_label = Text("Scale: 2x", font_size=20, color=YELLOW)
        self.place_in_area(scale_label, "E3", "E4", scale_factor=1.2)
        
        self.play(FadeIn(highlight_dot))
        self.play(Write(scale_label))
        
        self.wait(2)
        self.lecture[2].set_color(WHITE)

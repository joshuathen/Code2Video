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
        lecture_lines = [
            "Universal constants allow us to model complex chaotic systems.",
            "We can predict weather and design aircraft despite turbulence.",
            "Mathematics provides structure to the most chaotic flows."
        ]
        self.setup_layout("Synthesis: Predictability in Unpredictability", lecture_lines)

        # Colors
        chaos_color = "#ADD8E6"
        math_color = "#FFFF00"
        arrow_color = "#FFD700"
        text_color = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Chaotic Swirl
        swirl_func = lambda t: np.array([
            0.5 * t * np.cos(8 * t) + 0.1 * np.sin(20 * t),
            0.5 * t * np.sin(8 * t) + 0.1 * np.cos(25 * t),
            0
        ])
        swirl = ParametricFunction(swirl_func, t_range=[0, 3], color=chaos_color)
        swirl_label = Text("Real-world Chaos", font_size=20, color=chaos_color)
        
        # Aircraft Asset integration (Issue 20)
        aircraft = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/aircraft.svg").set_color(WHITE)
        
        self.place_in_area(swirl, "A1", "C3", scale_factor=0.8)
        self.place_at_grid(aircraft, "B2", scale_factor=0.4) # Placing aircraft inside the swirl
        # Issue 31: Use area positioning for swirl_label
        self.place_in_area(swirl_label, "D1", "D3", scale_factor=0.7)

        self.play(self.lecture[0].animate.set_color(chaos_color))
        self.play(Create(swirl), FadeIn(aircraft), Write(swirl_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Mathematical Predictability Graph (-5/3 slope)
        axes = Axes(
            x_range=[0.1, 10, 1],
            y_range=[0.1, 10, 1],
            axis_config={"include_tip": False, "stroke_width": 2},
            x_length=2.5,
            y_length=2.5
        ).set_color(WHITE)
        
        # Log-log style line: y = k * x^(-5/3)
        graph = axes.plot(lambda x: 5 * (x + 0.5)**(-5/3), x_range=[0.1, 8], color=math_color)
        math_label = Text("Mathematical Predictability", font_size=20, color=math_color)
        
        graph_group = VGroup(axes, graph)
        self.place_in_area(graph_group, "A4", "C6", scale_factor=0.8)
        # Issue 32: Use area positioning for math_label
        self.place_in_area(math_label, "D4", "D6", scale_factor=0.7)

        self.play(self.lecture[1].animate.set_color(math_color))
        self.play(Create(axes), Create(graph), Write(math_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Golden Arrow and Connecting Label
        
        # Create an arrow linking the areas
        link_arrow = Arrow(
            start=self.grid["B2"] + RIGHT * 1.5, 
            end=self.grid["B5"] + LEFT * 1.5, 
            color=arrow_color,
            stroke_width=6
        )
        link_label = Text("Universal Laws of Turbulence", font_size=24, color=text_color)
        
        # Positioning arrow and label in lower grid rows
        self.place_in_area(link_arrow, "E2", "E5", scale_factor=1.0)
        # Issue 33: Fix scale factor for link_label
        self.place_in_area(link_label, "F2", "F5", scale_factor=0.7)

        self.play(self.lecture[2].animate.set_color(arrow_color))
        self.play(GrowArrow(link_arrow))
        self.play(Write(link_label))
        self.wait(2)

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
        # Data from storyboard and outline
        title = "Application: Solving Systems of Differential Equations"
        lines = [
            "This provides the general solution for linear systems.",
            "The matrix exponential acts as the system's propagator.",
            "Watch the population levels evolve over time."
        ]
        
        self.setup_layout(title, lines)
        
        # Initial color setup for lecture highlighting
        self.lecture.set_color(GRAY)
        
        # Colors
        COLOR_RABBIT = "#00FF00"
        COLOR_FOX = "#FF8C00"
        COLOR_SOLUTION = "#FFFFFF"
        
        # Assets
        RABBIT_PATH = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/rabbit.svg"
        FOX_PATH = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/fox.svg"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # System of differential equations
        # Using Text to avoid LaTeX dependency issues as noted in previous version
        system_eq = Text("dx/dt = Ax", font_size=32, color=WHITE)
        self.place_at_grid(system_eq, "A3", scale_factor=1.0)
        
        self.play(Write(system_eq))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_SOLUTION))
        
        # Reveal solution as the 'Master Key'
        solution_eq = Text("x(t) = exp(At)x(0)", font_size=32, color=COLOR_SOLUTION)
        self.place_at_grid(solution_eq, "B3", scale_factor=1.0)
        
        master_key_label = Text("The Master Key", font_size=24, color=COLOR_SOLUTION)
        self.place_in_area(master_key_label, "B5", "B6", scale_factor=0.7)
        
        self.play(
            FadeIn(solution_eq, shift=UP),
            Write(master_key_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Setup population graph
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 6, 2],
            x_length=4.5,
            y_length=2.5,
            axis_config={"include_tip": True, "color": GRAY},
            tips=False
        ).add_coordinates(label_constructor=Text)
        self.place_in_area(axes, "D1", "F6", scale_factor=1.0)
        
        # Labels for axes
        labels = axes.get_axis_labels(
            x_label=Text("Time (t)", font_size=16),
            y_label=Text("Pop.", font_size=16)
        )
        
        self.play(Create(axes), Write(labels))
        
        # Load Assets
        rabbit_icon = SVGMobject(RABBIT_PATH).set_color(COLOR_RABBIT).scale(0.3)
        fox_icon = SVGMobject(FOX_PATH).set_color(COLOR_FOX).scale(0.3)
        
        # Labels with Icons
        rabbit_text = Text("Rabbits", font_size=18, color=COLOR_RABBIT)
        rabbit_label = VGroup(rabbit_icon, rabbit_text).arrange(RIGHT, buff=0.1)
        self.place_at_grid(rabbit_label, "C2", scale_factor=1.0)
        
        fox_text = Text("Foxes", font_size=18, color=COLOR_FOX)
        fox_label = VGroup(fox_icon, fox_text).arrange(RIGHT, buff=0.1)
        self.place_at_grid(fox_label, "C5", scale_factor=1.0)
        
        self.play(
            FadeIn(rabbit_label),
            FadeIn(fox_label)
        )
        
        # Population curves
        # Rabbit: 3 + 1.5*sin(t)
        # Fox: 3 + 1.5*cos(t)
        rabbit_curve = axes.plot(lambda t: 3 + 1.5 * np.sin(t), color=COLOR_RABBIT, x_range=[0, 10])
        fox_curve = axes.plot(lambda t: 3 + 1.5 * np.cos(t), color=COLOR_FOX, x_range=[0, 10])
        
        self.play(
            Create(rabbit_curve),
            Create(fox_curve),
            run_time=4
        )
        self.wait(2)

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
        lecture_lines = [
            "Meet Leo, sunbathing on a warm, desert rock.",
            "The rock's temperature varies by position and time.",
            "Ordinary equations track change over just one variable.",
            "But PDEs track change across space and time simultaneously.",
            "They model how heat flows through the entire rock."
        ]
        self.setup_layout("The Hook: Leo the Lizard and the Hot Rock", lecture_lines)

        # Colors
        ROCK_GREY = "#808080"
        HEAT_START = "#FF4500"
        HEAT_END = "#FFD700"
        RIPPLE_COLOR = "#FF0000"
        HIGHLIGHT_COLOR = "#FFFF00"
        ODE_BLUE = "#58C4DD"
        PDE_TEAL = "#5CD0B3"
        LIZARD_GREEN = "#32CD32"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Rock rectangle
        rock = Rectangle(width=4, height=3, fill_color=ROCK_GREY, fill_opacity=1.0, stroke_color=WHITE)
        self.place_in_area(rock, 'A2', 'F5')
        
        # Leo SVG Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/lizard.svg]
        try:
            leo = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lizard.svg")
            leo.set_color(LIZARD_GREEN)
        except Exception:
            # Fallback if asset fails to load
            leo = Triangle(fill_color=LIZARD_GREEN, fill_opacity=1.0).scale(0.3)
            
        self.place_in_area(leo, 'C3', 'D4', scale_factor=0.7)
        
        self.play(FadeIn(rock), FadeIn(leo))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Color the rock with a gradient
        self.play(rock.animate.set_fill(color=[HEAT_START, HEAT_END]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Dim the background to show the graph
        self.play(rock.animate.set_opacity(0.2), leo.animate.set_opacity(0.2))
        
        axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 2, 1],
            x_length=3.5, y_length=2.5,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, 'B2', 'E5')
        
        ode_graph = axes.plot(lambda x: 1 + 0.4 * np.sin(2*x), x_range=[0, 3.5], color=ODE_BLUE)
        ode_label = Text("T(t)", font_size=20, color=ODE_BLUE)
        # Fix for Issue 21: Position ode_label at A5 with scale_factor 0.8
        self.place_at_grid(ode_label, 'A5', scale_factor=0.8)
        
        self.play(Create(axes), Create(ode_graph), Write(ode_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # 2D Grid
        grid_2d = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=3.5, y_length=3.5,
            background_line_style={"stroke_color": ODE_BLUE, "stroke_opacity": 0.5}
        )
        self.place_in_area(grid_2d, 'B2', 'E5')
        
        pde_label = Text("T(x, y, t)", font_size=20, color=PDE_TEAL)
        # Fix for Issue 22: Position pde_label at A5 with scale_factor 0.8
        self.place_at_grid(pde_label, 'A5', scale_factor=0.8)
        
        self.play(
            FadeOut(axes), FadeOut(ode_graph), FadeOut(ode_label),
            FadeIn(grid_2d), Write(pde_label)
        )
        # Pulse color
        self.play(grid_2d.animate.set_color(PDE_TEAL), run_time=1)
        self.play(grid_2d.animate.set_color(ODE_BLUE), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Show heat ripples
        self.play(FadeOut(pde_label), FadeOut(grid_2d), rock.animate.set_opacity(1.0), leo.animate.set_opacity(1.0))
        
        # Create ripples
        ripple1 = Circle(radius=0.1, color=RIPPLE_COLOR, stroke_width=4)
        ripple2 = Circle(radius=0.1, color=RIPPLE_COLOR, stroke_width=4)
        ripple3 = Circle(radius=0.1, color=RIPPLE_COLOR, stroke_width=4)
        self.place_in_area(ripple1, 'C3', 'D4')
        self.place_in_area(ripple2, 'C3', 'D4')
        self.place_in_area(ripple3, 'C3', 'D4')
        
        self.play(
            LaggedStart(
                ripple1.animate.scale(15).set_stroke(opacity=0),
                ripple2.animate.scale(15).set_stroke(opacity=0),
                ripple3.animate.scale(15).set_stroke(opacity=0),
                lag_ratio=0.5,
                run_time=2.5,
                rate_func=linear
            )
        )
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)

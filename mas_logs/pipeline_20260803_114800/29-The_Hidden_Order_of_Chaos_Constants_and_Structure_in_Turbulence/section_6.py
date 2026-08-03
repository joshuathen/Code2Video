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
            "Turbulence follows universal laws across all physical scales.",
            "The -5/3 power law appears throughout our universe.",
            "It describes atmospheric winds and vast galactic nebulae.",
            "Chaotic motion masks a deep, predictable mathematical structure.",
            "We find ordered constants within the heart of chaos."
        ]
        self.setup_layout("Summary: The Universality of Turbulence", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Split-screen: Coffee vs Nebula
        self.lecture[0].set_color(YELLOW)
        
        # Representative coffee swirls
        coffee_cup = Circle(radius=1.0, color=WHITE, fill_color="#4B3621", fill_opacity=1)
        swirl_coords = [[0.2, 0.2, 0], [-0.3, 0.4, 0], [0.1, -0.5, 0], [-0.4, -0.2, 0]]
        swirls = VGroup(*[
            Arc(radius=0.4, start_angle=i*PI/2, angle=PI, color=WHITE).move_to(coffee_cup.get_center() + np.array(coord))
            for i, coord in enumerate(swirl_coords)
        ]).set_stroke(width=2, opacity=0.6)
        
        coffee_group = VGroup(coffee_cup, swirls)
        # Issue 38 Fix: place_in_area A1-B2, scale 0.6
        self.place_in_area(coffee_group, 'A1', 'B2', scale_factor=0.6)
        
        coffee_label = Text("Coffee Swirls", font_size=18, color=WHITE)
        # Issue 38 Fix: place_at_grid C1, scale 0.8
        self.place_at_grid(coffee_label, 'C1', scale_factor=0.8)
        
        # Representative nebula
        nebula_bg = Rectangle(width=2.5, height=2.5, fill_color=PURPLE_E, fill_opacity=0.3, stroke_width=0)
        star_coords = [
            [0.5, 0.8, 0], [-0.7, 0.3, 0], [0.2, -0.9, 0], [-0.4, -0.5, 0], [0.9, -0.2, 0],
            [-0.1, 0.1, 0], [0.6, 0.4, 0], [-0.8, -0.7, 0]
        ]
        stars = VGroup(*[
            Dot(point=nebula_bg.get_center() + np.array(coord), radius=0.03, color=WHITE) 
            for coord in star_coords
        ])
        nebula_group = VGroup(nebula_bg, stars)
        # Issue 38 Fix: place_in_area A5-B6, scale 0.6
        self.place_in_area(nebula_group, 'A5', 'B6', scale_factor=0.6)
        
        nebula_label = Text("Galactic Nebula", font_size=18, color=WHITE)
        # Issue 38 Fix: place_at_grid C6, scale 0.8
        self.place_at_grid(nebula_label, 'C6', scale_factor=0.8)
        
        self.play(
            FadeIn(coffee_group), Write(coffee_label),
            FadeIn(nebula_group), Write(nebula_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Superimpose the "-5/3 Law" log-log graph
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=4.0,
            y_length=3.0,
            axis_config={"color": WHITE},
            tips=False
        )
        # Issue 37 Fix: place_in_area C2-F5
        self.place_in_area(axes, 'C2', 'F5')
        
        # Log-log line with -5/3 slope
        graph_line = Line(
            axes.c2p(1, 5),
            axes.c2p(5, 1),
            color=YELLOW,
            stroke_width=5
        )
        graph_label = MathTex(r"E(k) \sim k^{-5/3}", color=YELLOW, font_size=32)
        # Issue 39 Fix: place_at_grid A3, scale 1.2
        self.place_at_grid(graph_label, 'A3', scale_factor=1.2)
        
        self.play(
            coffee_group.animate.set_opacity(0.4),
            nebula_group.animate.set_opacity(0.4),
            coffee_label.animate.set_opacity(0.4),
            nebula_label.animate.set_opacity(0.4),
            Create(axes),
            Create(graph_line),
            Write(graph_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # scales: atmospheric vs galactic
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(
            Indicate(coffee_label, color=BLUE_B), 
            Indicate(nebula_label, color=PURPLE_B)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Predictable structure
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        self.play(graph_line.animate.set_stroke(width=10, color=GOLD), run_time=1)
        self.wait(0.5)
        self.play(graph_line.animate.set_stroke(width=5, color=YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Final message
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        final_msg = Text("Structured Chaos:\nThe Universal Code of Nature", font_size=32, color=WHITE, weight=BOLD)
        msg_box = SurroundingRectangle(final_msg, buff=0.5, color=WHITE, fill_color=BLACK, fill_opacity=0.9)
        final_group = VGroup(msg_box, final_msg)
        
        self.play(
            FadeOut(axes), FadeOut(graph_line), FadeOut(graph_label),
            FadeOut(coffee_group), FadeOut(nebula_group),
            FadeOut(coffee_label), FadeOut(nebula_label)
        )
        
        self.place_in_area(final_group, 'B1', 'E6')
        self.play(FadeIn(msg_box), Write(final_msg))
        self.wait(3)

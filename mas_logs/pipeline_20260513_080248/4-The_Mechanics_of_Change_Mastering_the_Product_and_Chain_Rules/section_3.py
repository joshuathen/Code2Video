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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'The chain rule differentiates nested, or composed, functions.', 
            'Visualize an inner gear driving a larger outer gear.', 
            'Total speed is the product of individual rates.', 
            'Multiply the outer derivative by the inner derivative.', 
            "Like a snail's speed relative to the moving ground."
        ]
        self.setup_layout("The Chain Rule: The Nested Gear System", lecture_lines)
        
        # Colors
        INNER_COLOR = "#00FF00" # Green
        OUTER_COLOR = "#FF00FF" # Magenta
        TEXT_COLOR = "#FFFFFF"  # White

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Composition text: f(g(x))
        comp_f = Text("f(", color=OUTER_COLOR)
        comp_g = Text("g(x)", color=INNER_COLOR)
        comp_close = Text(")", color=OUTER_COLOR)
        composition = VGroup(comp_f, comp_g, comp_close).arrange(RIGHT, buff=0.1)
        # Fix Issue 29: place_in_area(composition, 'A3', 'B4', scale_factor=0.9)
        self.place_in_area(composition, "A3", "B4", scale_factor=0.9)
        
        self.play(Write(composition))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create Gears
        # Inner Gear g
        inner_gear = Star(n=12, inner_radius=0.6, outer_radius=0.8, color=INNER_COLOR)
        inner_gear_label = Text("g", color=INNER_COLOR)
        inner_gear_group = VGroup(inner_gear, inner_gear_label)
        self.place_at_grid(inner_gear_group, "C2", scale_factor=0.6)
        
        # Outer Gear f
        outer_gear = Star(n=20, inner_radius=1.0, outer_radius=1.3, color=OUTER_COLOR)
        outer_gear_label = Text("f", color=OUTER_COLOR)
        outer_gear_group = VGroup(outer_gear, outer_gear_label)
        self.place_at_grid(outer_gear_group, "C4", scale_factor=0.6)
        
        # Visual connector
        connector = Line(inner_gear_group.get_right(), outer_gear_group.get_left(), color=WHITE)

        self.play(
            Create(inner_gear_group),
            Create(outer_gear_group),
            Create(connector)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Rate labels
        g_prime = Text("g'(x)", color=INNER_COLOR, font_size=24)
        f_prime = Text("f'(g)", color=OUTER_COLOR, font_size=24)
        self.place_at_grid(g_prime, "D2", scale_factor=1.0)
        self.place_at_grid(f_prime, "D4", scale_factor=1.0)

        # Rotation animation
        self.play(
            Rotate(inner_gear, angle=2*PI, run_time=3),
            Rotate(outer_gear, angle=PI, run_time=3), # Outer moves slower/differently
            FadeIn(g_prime),
            FadeIn(f_prime)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Formula: [f(g(x))]' = f'(g(x)) * g'(x)
        # Constructing part by part for coloring
        part1 = Text("[f(g(x))]' =", color=TEXT_COLOR, font_size=32)
        part2 = Text("f'(g(x))", color=OUTER_COLOR, font_size=32)
        part3 = Text(" \u00b7 ", color=TEXT_COLOR, font_size=32) # Dot
        part4 = Text("g'(x)", color=INNER_COLOR, font_size=32)
        
        formula = VGroup(part1, part2, part3, part4).arrange(RIGHT, buff=0.2)
        # Fix Issue 30: place_in_area(formula, 'E2', 'E5', scale_factor=0.65)
        self.place_in_area(formula, "E2", "E5", scale_factor=0.65)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Snail Analogy
        train = Rectangle(height=0.8, width=2.0, color=BLUE, fill_opacity=0.3)
        train_text = Text("Train", font_size=20, color=BLUE)
        train_group = VGroup(train, train_text)
        
        snail = Circle(radius=0.15, color=INNER_COLOR, fill_opacity=1.0)
        snail_text = Text("Snail", font_size=18, color=INNER_COLOR)
        snail_group = VGroup(snail, snail_text).arrange(UP, buff=0.1)
        
        # Fix Issue 28: train_group at F2, snail_group at F4
        self.place_at_grid(train_group, "F2", scale_factor=0.8)
        self.place_at_grid(snail_group, "F4", scale_factor=0.8)
        
        ground = Line(self.grid["F1"] + LEFT*0.5, self.grid["F6"] + RIGHT*0.5, color=WHITE)
        
        self.play(
            FadeIn(train_group),
            FadeIn(snail_group),
            Create(ground)
        )
        
        # Moving train and snail
        self.play(
            train_group.animate.shift(RIGHT * 1.5),
            snail_group.animate.shift(RIGHT * 2.0), # Snail moves relative to train
            run_time=3,
            rate_func=linear
        )
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)

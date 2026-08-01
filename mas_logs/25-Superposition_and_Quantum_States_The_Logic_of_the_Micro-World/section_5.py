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
        # Setup titles and lecture lines
        lecture_lines = [
            "A cat's fate depends on a single quantum event.",
            "Mathematically, the cat is both alive and dead.",
            "This blurred existence remains until the box is opened.",
            "Opening the lid forces the system to choose reality.",
            "The cat collapses into a single, definite state."
        ]
        self.setup_layout("The Thought Experiment: Schrödinger's Cat", lecture_lines)

        # Define Colors
        COLOR_BOX = "#FFFFFF"
        COLOR_EQUATION = "#FFFFFF"
        COLOR_COLLAPSE = "#00FF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create Box
        box_body = Rectangle(width=4, height=3, color=COLOR_BOX)
        # Fix Issue 44: Positioning box lower
        self.place_in_area(box_body, "D2", "F5", scale_factor=0.8)
        
        lid = Line(box_body.get_corner(UL), box_body.get_corner(UR), color=COLOR_BOX, stroke_width=8)
        
        # Create Cat Silhouette (Standing)
        cat_alive = VGroup(
            Ellipse(width=0.6, height=0.4, color=WHITE, fill_opacity=0.3), # Body
            Circle(radius=0.2, color=WHITE, fill_opacity=0.3).shift(UP*0.3 + RIGHT*0.2), # Head
            Triangle(color=WHITE, fill_opacity=0.3).scale(0.1).shift(UP*0.5 + RIGHT*0.1), # Ear 1
            Triangle(color=WHITE, fill_opacity=0.3).scale(0.1).shift(UP*0.5 + RIGHT*0.3), # Ear 2
        ).move_to(box_body.get_center() + LEFT*0.5)
        
        # Create Cat Silhouette (Dead/Lying)
        cat_dead = VGroup(
            Ellipse(width=0.6, height=0.25, color=WHITE, fill_opacity=0.3), # Body flat
            Circle(radius=0.15, color=WHITE, fill_opacity=0.3).shift(RIGHT*0.4), # Head to side
        ).move_to(box_body.get_center() + LEFT*0.5).rotate(10*DEGREES)

        # Radioactive Atom
        atom_core = Circle(radius=0.1, color=GREEN, fill_opacity=1).move_to(box_body.get_center() + RIGHT*1.0)
        atom_orbit1 = Ellipse(width=0.5, height=0.2, color=GREEN).move_to(atom_core)
        atom_orbit2 = atom_orbit1.copy().rotate(60*DEGREES)
        atom = VGroup(atom_core, atom_orbit1, atom_orbit2)

        self.play(Create(box_body), Create(lid), FadeIn(cat_alive), FadeIn(atom))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Equation: |Ψ⟩ = 1/√2(|Alive⟩ + |Dead⟩)
        psi_symbol = Text("|Ψ⟩", font_size=24)
        eq_sign = Text("=", font_size=24)
        coeff = Text("1/√2", font_size=24)
        bracket_l = Text("(", font_size=24)
        alive_term = Text("|Alive⟩", font_size=24)
        plus_sign = Text("+", font_size=24)
        dead_term = Text("|Dead⟩", font_size=24)
        bracket_r = Text(")", font_size=24)

        equation = VGroup(psi_symbol, eq_sign, coeff, bracket_l, alive_term, plus_sign, dead_term, bracket_r).arrange(RIGHT, buff=0.15)
        # Fix Issue 43: Placing equation lower and smaller
        self.place_in_area(equation, "B1", "B6", scale_factor=0.8)
        
        self.play(FadeIn(equation))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Flicker superposition
        for _ in range(3):
            self.play(FadeOut(cat_alive), FadeIn(cat_dead), run_time=0.2)
            self.play(FadeIn(cat_alive), FadeOut(cat_dead), run_time=0.2)
        
        # Keep both partially visible to represent superposition
        cat_dead.set_opacity(0.3)
        self.add(cat_dead)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Open Lid
        pivot = lid.get_start()
        self.play(Rotate(lid, angle=PI/2, about_point=pivot))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_COLLAPSE)

        # Equation Collapse
        new_equation = VGroup(
            Text("|Ψ⟩", font_size=24, color=COLOR_COLLAPSE),
            Text("=", font_size=24, color=COLOR_COLLAPSE),
            Text("|Alive⟩", font_size=24, color=COLOR_COLLAPSE)
        ).arrange(RIGHT, buff=0.15)
        # Fix Issue 45: Centering final equation
        self.place_in_area(new_equation, "B2", "B5", scale_factor=0.8)

        self.play(
            ReplacementTransform(equation, new_equation),
            FadeOut(cat_dead),
            cat_alive.animate.set_color(COLOR_COLLAPSE).set_opacity(1.0).scale(1.1)
        )
        self.play(cat_alive.animate.scale(1/1.1))
        self.wait(2)

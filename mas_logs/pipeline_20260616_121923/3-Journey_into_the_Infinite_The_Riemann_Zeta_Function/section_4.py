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
        # Setup the layout with specific section 4 content
        lines = [
            'We can extend Zeta beyond its original limits.',
            'Analytic continuation stretches the function across complex space.',
            'This reveals hidden values in the negative plane.',
            'It leads to the counter-intuitive sum of natural numbers.',
            "The 'impossible' result equals negative one twelfth."
        ]
        self.setup_layout("Analytic Continuation: Crossing the Barrier", lines)
        
        # Colors
        BLUE_REGION = "#4682B4"
        PURPLE_MESH = "#BA55D3"
        POLE_RED = "#FF0000"
        ZETA_GREEN = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Create a background grid for context
        bg_plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.2}
        )
        self.place_in_area(bg_plane, "A1", "F6", scale_factor=0.6)
        self.add(bg_plane)

        # Region Re(s) > 1
        blue_box = Rectangle(width=2, height=4, fill_color=BLUE_REGION, fill_opacity=0.5, stroke_width=0)
        self.place_in_area(blue_box, "A4", "F6", scale_factor=0.8) # Area covers right side
        
        # Fixed: Replaced MathTex with Text to avoid LaTeX dependency error
        zeta_formula = Text("ζ(s) = Σ 1/nˢ", color=WHITE)
        # Issue 45 Fix: Line 80: self.place_in_area(zeta_formula, 'B5', 'C6', scale_factor=0.6)
        self.place_in_area(zeta_formula, 'B5', 'C6', scale_factor=0.6)
        
        self.lecture[0].set_color(BLUE_REGION)
        self.play(FadeIn(blue_box), Write(zeta_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        mesh = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            axis_config={"stroke_opacity": 0},
            background_line_style={"stroke_color": PURPLE_MESH, "stroke_width": 2, "stroke_opacity": 0.8}
        )
        self.place_in_area(mesh, "A1", "F6", scale_factor=0.6)
        
        mesh.save_state()
        mesh.stretch_to_fit_width(0.1)
        self.place_at_grid(mesh, "D4", scale_factor=1.0) # Start from s=1 boundary
        
        self.lecture[1].set_color(PURPLE_MESH)
        self.play(
            Restore(mesh),
            blue_box.animate.set_fill(opacity=0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        pole_dot = Circle(radius=0.1, color=POLE_RED, fill_opacity=1)
        cross = VGroup(
            Line(UP+LEFT, DOWN+RIGHT),
            Line(UP+RIGHT, DOWN+LEFT)
        ).scale(0.1).set_color(WHITE)
        pole_group = VGroup(pole_dot, cross)
        self.place_at_grid(pole_group, "D4", scale_factor=1.0) # s=1
        
        pole_label = Text("Pole at s=1", font_size=18, color=POLE_RED)
        self.place_at_grid(pole_label, "E4", scale_factor=0.8)
        
        self.lecture[2].set_color(POLE_RED)
        self.play(Create(pole_group), FadeIn(pole_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Issue 33 Fix: [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/magnifier.svg]
        mag_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/magnifier.svg").set_color(WHITE)
        self.place_at_grid(mag_icon, "D2", scale_factor=0.6) # s=-1
        
        # Fixed: Replaced MathTex with Text to avoid LaTeX dependency error
        nat_sum = Text("1 + 2 + 3 + ...", color=WHITE)
        # Issue 43 Fix: Line 129: self.place_in_area(nat_sum, 'A1', 'B3', scale_factor=0.7)
        self.place_in_area(nat_sum, 'A1', 'B3', scale_factor=0.7)
        
        self.lecture[3].set_color(WHITE)
        self.play(Create(mag_icon), Write(nat_sum))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Fixed: Replaced MathTex with Text to avoid LaTeX dependency error
        result_formula = Text("ζ(-1) = -1/12", color=ZETA_GREEN)
        # Issue 44 Fix: Line 138: self.place_in_area(result_formula, 'C1', 'C3', scale_factor=0.8)
        self.place_in_area(result_formula, 'C1', 'C3', scale_factor=0.8)
        
        self.lecture[4].set_color(ZETA_GREEN)
        
        pulse_mesh = mesh.copy().set_stroke(width=4)
        self.play(
            Write(result_formula),
            FadeIn(pulse_mesh),
            pole_label.animate.set_opacity(0.5)
        )
        self.play(
            pulse_mesh.animate.set_stroke(opacity=0).scale(1.1),
            run_time=1,
            rate_func=lambda t: t # linear
        )
        self.remove(pulse_mesh)
        self.wait(2)

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
        # Mandatory call to setup_layout
        self.setup_layout(
            "The Mystery of the Five Constants",
            [
                "This equation links the five most important mathematical constants.",
                "It unites zero and one, the foundations of arithmetic.",
                "Pi brings in the geometry of the circle.",
                "The imaginary unit i introduces rotation into the mix.",
                "Base e connects them through the power of growth."
            ]
        )
        
        # Initial state: Dim lecture lines to highlight them during narration
        self.lecture.set_opacity(0.3)

        # Formula components: e^(πi) + 1 = 0
        # Splitting for granular color and opacity control
        f_e = Text("e")
        f_sup_open = Text("^(", font_size=20)
        f_pi = Text("π")
        f_i = Text("i")
        f_sup_close = Text(")", font_size=20)
        f_plus = Text(" + ")
        f_one = Text("1")
        f_eq = Text(" = ")
        f_zero = Text("0")
        
        formula_group = VGroup(f_e, f_sup_open, f_pi, f_i, f_sup_close, f_plus, f_one, f_eq, f_zero)
        formula_group.arrange(RIGHT, buff=0.08, aligned_edge=DOWN)
        
        # Manually adjust superscript positioning
        for part in [f_sup_open, f_pi, f_i, f_sup_close]:
            part.shift(UP * 0.25)

        # Issue 26: Fix formula placement and scale (B2 to C5 area)
        self.place_in_area(formula_group, 'B2', 'C5', scale_factor=1.1)

        # === Animation for Lecture Line 1 ===
        # "This equation links the five most important mathematical constants."
        self.play(
            Write(formula_group),
            self.lecture[0].animate.set_opacity(1).set_color(WHITE),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "It unites zero and one, the foundations of arithmetic."
        # Highlight '0' in red (#FF0000) and '1' in green (#00FF00)
        # Indices: 0:e, 1:^(, 2:π, 3:i, 4:), 5:+, 6:1, 7:=, 8:0
        self.play(
            self.lecture[1].animate.set_opacity(1).set_color(WHITE),
            formula_group[8].animate.set_color("#FF0000"), # Constant 0
            formula_group[6].animate.set_color("#00FF00"), # Constant 1
            formula_group[0:6].animate.set_opacity(0.3),
            formula_group[7].animate.set_opacity(0.3),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Pi brings in the geometry of the circle."
        # Highlight 'pi' in yellow (#FFFF00) and draw a unit circle outline (#555555) nearby.
        # Issue 27: Plane at D3-F4 area
        plane = ComplexPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_opacity": 0.2}
        )
        self.place_in_area(plane, 'D3', 'F4', scale_factor=0.8)
        
        circle = Circle(radius=plane.get_x_unit_size(), color="#555555")
        circle.move_to(plane.get_origin())

        self.play(
            self.lecture[2].animate.set_opacity(1).set_color("#FFFF00"),
            formula_group[2].animate.set_color("#FFFF00").set_opacity(1), # pi
            formula_group[0:2].animate.set_opacity(0.3),
            formula_group[3:9].animate.set_opacity(0.3),
            Create(plane),
            Create(circle),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The imaginary unit i introduces rotation into the mix."
        # Highlight 'i' in cyan (#00FFFF) and show a 90-degree arc arrow (#00FFFF)
        arc = ArcBetweenPoints(
            plane.n2p(1), 
            plane.n2p(1j), 
            radius=plane.get_x_unit_size(), 
            color="#00FFFF"
        )
        arc.add_tip()

        self.play(
            self.lecture[3].animate.set_opacity(1).set_color("#00FFFF"),
            formula_group[3].animate.set_color("#00FFFF").set_opacity(1), # i
            formula_group[0:3].animate.set_opacity(0.3),
            formula_group[4:9].animate.set_opacity(0.3),
            Create(arc),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Base e connects them through the power of growth."
        # Highlight 'e' in magenta (#FF00FF) with a horizontal growth arrow (#FF00FF)
        growth_arrow = Arrow(
            start=plane.n2p(1), 
            end=plane.n2p(2), 
            color="#FF00FF", 
            buff=0
        )

        self.play(
            self.lecture[4].animate.set_opacity(1).set_color("#FF00FF"),
            formula_group[0].animate.set_color("#FF00FF").set_opacity(1), # e
            formula_group[1:9].animate.set_opacity(0.3),
            GrowArrow(growth_arrow),
            run_time=1.5
        )
        self.wait(1)
        
        # Final sequence: all 5 symbols flash together
        self.play(
            formula_group.animate.set_opacity(1),
            formula_group[0].animate.set_color("#FF00FF"), # e
            formula_group[2].animate.set_color("#FFFF00"), # pi
            formula_group[3].animate.set_color("#00FFFF"), # i
            formula_group[6].animate.set_color("#00FF00"), # 1
            formula_group[8].animate.set_color("#FF0000"), # 0
            Flash(formula_group[0], color="#FF00FF"),
            Flash(formula_group[2], color="#FFFF00"),
            Flash(formula_group[3], color="#00FFFF"),
            Flash(formula_group[6], color="#00FF00"),
            Flash(formula_group[8], color="#FF0000"),
            run_time=2
        )
        self.wait(3)

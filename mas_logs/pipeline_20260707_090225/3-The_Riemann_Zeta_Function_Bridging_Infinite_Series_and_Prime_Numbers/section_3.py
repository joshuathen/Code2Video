from manim import *

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
        # Define lecture lines
        lecture_lines = [
            "We define the Riemann Zeta function with this sum.",
            "At s equals two, the sum is pi-squared over six.",
            "This connects simple integers to the geometry of circles."
        ]
        self.setup_layout("Defining the Riemann Zeta Function", lecture_lines)

        # Pre-dim lecture lines
        for line in self.lecture:
            line.set_color(GRAY_D)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Label: Infinite Series Representation
        representation_label = Text("Infinite Series Representation", font_size=24, color="#FFFFFF")
        self.place_in_area(representation_label, 'A1', 'A6', scale_factor=0.6)
        
        # Zeta formula definition
        # Using Text instead of MathTex for stability
        zeta_formula = Text("ζ(s) = Σ 1/nˢ = 1 + 1/2ˢ + 1/3ˢ + ...", font_size=32, color="#FFFFFF")
        self.place_in_area(zeta_formula, 'B1', 'C6', scale_factor=0.8)
        
        self.play(Write(representation_label))
        self.play(FadeIn(zeta_formula, shift=UP))
        self.wait(1)

        # Mapping visualization: s -> Box [ζ] -> value
        box = Rectangle(width=1.0, height=1.0, color=WHITE)
        zeta_sym = Text("ζ", color=WHITE)
        zeta_box = VGroup(box, zeta_sym)
        self.place_at_grid(zeta_box, 'B3', scale_factor=0.7)

        input_s = Text("s", color="#00FF00")
        self.place_at_grid(input_s, 'B1', scale_factor=0.8)
        
        output_val = Text("Value", color="#00FFFF")
        self.place_at_grid(output_val, 'B5', scale_factor=0.8)

        # Shift formula out of the way for mapping demo if needed, 
        # but the prompt says zeta_formula is in B1-C6.
        self.play(FadeOut(zeta_formula))
        self.play(Create(zeta_box))
        self.play(input_s.animate.move_to(zeta_box.get_center()))
        self.play(FadeOut(input_s, scale=0.5), FadeIn(output_val, shift=RIGHT))
        self.wait(1)
        self.play(FadeOut(zeta_box), FadeOut(output_val), FadeOut(representation_label))

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(GRAY_D),
            self.lecture[1].animate.set_color("#FFFF00")
        )

        s2_formula = Text("ζ(2) = 1/1² + 1/2² + 1/3² + ... = π²/6", font_size=32, color="#FFFF00")
        self.place_in_area(s2_formula, 'D1', 'F6', scale_factor=0.8)
        
        self.play(Write(s2_formula))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(GRAY_D),
            self.lecture[2].animate.set_color("#FF00FF")
        )

        # Pink circle unrolling into a line
        circle = Circle(radius=0.6, color="#FF00FF")
        self.place_at_grid(circle, 'B3', scale_factor=1.0)
        
        # Calculate perimeter (2*pi*r)
        line_length = 2 * PI * 0.6
        unrolled_line = Line(LEFT * (line_length/2), RIGHT * (line_length/2), color="#FF00FF")
        self.place_at_grid(unrolled_line, 'B3', scale_factor=1.0)
        
        # Segment of length pi (half of the 2pi line)
        pi_segment = Line(LEFT * (line_length/4), RIGHT * (line_length/4), color="#FF0000", stroke_width=6)
        self.place_at_grid(pi_segment, 'B3', scale_factor=1.0)
        pi_segment.shift(UP * 0.2) # Slight offset to see clearly

        self.play(Create(circle))
        self.wait(0.5)
        self.play(ReplacementTransform(circle, unrolled_line))
        self.play(Create(pi_segment))
        
        # Gold bridge connecting sum to geometric representation
        bridge = ArcBetweenPoints(
            s2_formula.get_top(),
            unrolled_line.get_bottom(),
            angle=-TAU/4,
            color="#FFD700"
        )
        bridge_label = Text("The Bridge", font_size=20, color="#FFD700").next_to(bridge, RIGHT, buff=0.1)
        
        self.play(Create(bridge))
        self.play(Write(bridge_label))
        self.wait(2)

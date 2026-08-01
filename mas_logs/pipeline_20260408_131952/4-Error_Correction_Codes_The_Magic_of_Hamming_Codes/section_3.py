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
        # Setup the title and lecture lines
        lecture_lines = [
            'Hamming used overlapping circles to check data bits.',
            'Place four data bits in the overlapping regions.',
            'Add three parity bits to the outer circles.',
            'Every data bit is covered by multiple parity checks.',
            'This redundancy allows us to pinpoint errors.'
        ]
        self.setup_layout("The Hamming Logic: Overlapping Circles", lecture_lines)

        # Colors for the circles
        p1_color = "#FF5733" # Reddish
        p2_color = "#33FF57" # Greenish
        p4_color = "#3357FF" # Blueish

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Draw three overlapping circles
        c1 = Circle(radius=2.3, color=p1_color, stroke_width=4, fill_opacity=0.2)
        c2 = Circle(radius=2.3, color=p2_color, stroke_width=4, fill_opacity=0.2)
        c4 = Circle(radius=2.3, color=p4_color, stroke_width=4, fill_opacity=0.2)

        self.place_in_area(c1, "B3", "B4")
        self.place_at_grid(c2, "D2")
        self.place_at_grid(c4, "D5")

        self.play(Create(c1), Create(c2), Create(c4), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Data bits: D3, D5, D6, D7 (Using Text instead of MathTex to avoid LaTeX requirement)
        d3 = Text("D3", font_size=34, color=WHITE)
        d5 = Text("D5", font_size=34, color=WHITE)
        d6 = Text("D6", font_size=34, color=WHITE)
        d7 = Text("D7", font_size=34, color=WHITE)

        # Place data bits in intersections
        # Fixes for Issues 34, 35, 36: use specific grid points and scale 0.8
        self.place_at_grid(d3, 'C2', scale_factor=0.8) # P1 & P2
        self.place_at_grid(d5, 'C5', scale_factor=0.8) # P1 & P4
        self.place_at_grid(d6, 'E4', scale_factor=0.8) # P2 & P4
        self.place_at_grid(d7, 'D4', scale_factor=0.8) # All (Center)

        self.play(Write(VGroup(d3, d5, d6, d7)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Parity bits: P1, P2, P4 (Using Text instead of MathTex to avoid LaTeX requirement)
        p1_lbl = Text("P1", font_size=34, color=p1_color)
        p2_lbl = Text("P2", font_size=34, color=p2_color)
        p4_lbl = Text("P4", font_size=34, color=p4_color)

        # Place parity bits in outer petals
        self.place_in_area(p1_lbl, "A3", "A4", scale_factor=1.0) # Top
        self.place_in_area(p2_lbl, "E1", "E2", scale_factor=1.0) # Left
        self.place_in_area(p4_lbl, "E5", "E6", scale_factor=1.0) # Right

        self.play(Write(VGroup(p1_lbl, p2_lbl, p4_lbl)))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )

        # Highlight Circle P1: P1, D3, D5, D7
        self.play(c1.animate.set_stroke(width=10, opacity=1), run_time=0.8)
        self.play(Indicate(VGroup(p1_lbl, d3, d5, d7), color=p1_color))
        self.play(c1.animate.set_stroke(width=4, opacity=1), run_time=0.5)

        # Highlight Circle P2: P2, D3, D6, D7
        self.play(c2.animate.set_stroke(width=10, opacity=1), run_time=0.8)
        self.play(Indicate(VGroup(p2_lbl, d3, d6, d7), color=p2_color))
        self.play(c2.animate.set_stroke(width=4, opacity=1), run_time=0.5)

        # Highlight Circle P4: P4, D5, D6, D7
        self.play(c4.animate.set_stroke(width=10, opacity=1), run_time=0.8)
        self.play(Indicate(VGroup(p4_lbl, d5, d6, d7), color=p4_color))
        self.play(c4.animate.set_stroke(width=4, opacity=1), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Emphasize redundancy (bit 7)
        self.play(
            c1.animate.set_fill(opacity=0.4),
            c2.animate.set_fill(opacity=0.4),
            c4.animate.set_fill(opacity=0.4),
            run_time=1
        )
        self.play(Indicate(d7, scale_factor=2.0, color=YELLOW), run_time=1.5)
        self.wait(2)

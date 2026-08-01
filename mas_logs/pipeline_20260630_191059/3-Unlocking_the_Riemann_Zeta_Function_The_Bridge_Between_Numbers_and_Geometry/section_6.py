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

class Section6Scene(TeachingScene):
    def construct(self):
        # Fetching content from storyboard
        title = "The Riemann Hypothesis: The Critical Line"
        lecture_lines = [
            "We focus on the critical strip between zero and one.",
            "A yellow line marks exactly one half on the axis.",
            "Riemann conjectured all non-trivial zeros lie on this line.",
            "Like a tightrope walker, the function maintains perfect balance.",
            "Proving this hypothesis would unlock the secret of primes."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # The view zooms into the 'Critical Strip' between σ=0 and σ=1, highlighted in grey (#808080).
        self.lecture[0].set_color("#808080")
        
        # Strip from col 2 to col 4
        strip = Rectangle(width=2.0, height=5.0, color="#808080", fill_opacity=0.3, stroke_width=0)
        self.place_in_area(strip, "A2", "F4")
        
        # Labels for σ=0 and σ=1
        label0 = Text("σ=0", font_size=18, color="#808080")
        label1 = Text("σ=1", font_size=18, color="#808080")
        self.place_at_grid(label0, "F2")
        self.place_at_grid(label1, "F4")
        
        self.play(FadeIn(strip), Write(label0), Write(label1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A bright yellow (#FFFF00) vertical line is drawn at σ = 0.5, labeled as the 'Critical Line'.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        crit_line = Line(self.grid["A3"], self.grid["F3"], color="#FFFF00")
        label_half = Text("σ=0.5", font_size=18, color="#FFFF00")
        self.place_at_grid(label_half, "F3")
        
        line_label = Text("Critical Line", font_size=20, color="#FFFF00")
        # Fix for Issue 45: Scaled to 0.6
        self.place_at_grid(line_label, "A3", scale_factor=0.6)
        line_label.shift(UP * 0.3)
        
        self.play(Create(crit_line), Write(label_half), Write(line_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A sequence of blue dots (#0000FF) representing zeros appears, all perfectly aligned on the yellow line.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#0000FF") 
        
        # Create some zeros on the line (Col 3)
        zeros = VGroup(*[
            Dot(self.grid[f"{r}3"], color="#0000FF", radius=0.1) 
            for r in ["E", "D", "C", "B"]
        ])
        
        self.play(Create(zeros))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A tightrope walker icon [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/tightrope.svg]
        # moves along the yellow line, successfully crossing the gap between the zeros.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW) 
        
        # Fix for Issue 27: Integrated SVGMobject asset
        walker = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/tightrope.svg")
        walker.set_color(YELLOW)
        # Fix for Issue 46: Positioned at E3, scale 0.4
        self.place_at_grid(walker, "E3", scale_factor=0.4)
        
        self.add(walker)
        # Move walker from bottom to top along the critical line
        self.play(walker.animate.move_to(self.grid["A3"]), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The text 'Riemann Hypothesis' appears in bold gold (#FFD700) as the walker reaches the other side.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFD700")
        
        rh_text = Text("Riemann Hypothesis", color="#FFD700", weight=BOLD, font_size=28)
        # Fix for Issue 44: Replaced position and scale
        self.place_in_area(rh_text, "B5", "C6", scale_factor=0.6) 
        
        self.play(Write(rh_text))
        self.wait(2)

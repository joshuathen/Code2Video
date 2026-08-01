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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the title and lecture lines
        lecture_lines = [
            "Mathematics has five fundamental constants that define our world.",
            "Meet zero, one, e, pi, and the imaginary unit i.",
            "Separately they seem unrelated, but one formula unites them."
        ]
        self.setup_layout("The Gathering of Five Giants", lecture_lines)

        # Create symbols using Text instead of MathTex to bypass missing LaTeX installation
        # Variables like e and i are slanted to mimic mathematical style
        zero = Text("0", color="#FFFFFF")
        one = Text("1", color="#FFFFFF")
        e_symbol = Text("e", color="#FFD700", slant=ITALIC)
        pi = Text("π", color="#ADD8E6")
        i_unit = Text("i", color="#FF69B4", slant=ITALIC)

        # === Animation for Lecture Line 1 ===
        # Initial arrangement in a circle-like pattern
        self.place_at_grid(zero, "B3", scale_factor=2.0)
        self.place_at_grid(one, "B4", scale_factor=2.0)
        self.place_at_grid(e_symbol, "C5", scale_factor=2.0)
        self.place_at_grid(pi, "D4", scale_factor=2.0)
        self.place_at_grid(i_unit, "C2", scale_factor=2.0)

        # Display constants
        self.play(
            LaggedStart(
                FadeIn(zero), FadeIn(one), FadeIn(e_symbol), FadeIn(pi), FadeIn(i_unit),
                lag_ratio=0.2
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color change for lecture line
        self.play(self.lecture[1].animate.set_color(WHITE))

        # Labels for the symbols, positioned 'underneath' (one row below)
        label_0 = Text("Additive Identity", font_size=16, color="#FFFFFF")
        label_1 = Text("Multiplicative Identity", font_size=16, color="#FFFFFF")
        label_e = Text("Growth Base", font_size=16, color="#FFD700")
        label_pi = Text("Circle Constant", font_size=16, color="#ADD8E6")
        label_i = Text("Imaginary Unit", font_size=16, color="#FF69B4")

        # Position labels one grid unit below their respective symbols
        self.place_at_grid(label_0, "C3", scale_factor=1.0)
        self.place_at_grid(label_1, "C4", scale_factor=1.0)
        self.place_at_grid(label_e, "D5", scale_factor=1.0)
        self.place_at_grid(label_pi, "E4", scale_factor=1.0)
        self.place_at_grid(label_i, "D2", scale_factor=1.0)

        # Sequential scaling and labeling
        for mob, lab in [(zero, label_0), (one, label_1), (e_symbol, label_e), (pi, label_pi), (i_unit, label_i)]:
            self.play(
                mob.animate.scale(1.2),
                Write(lab),
                run_time=0.8
            )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color change for lecture line
        self.play(self.lecture[2].animate.set_color(WHITE))

        # Move symbols into a horizontal alignment (preparing for equation)
        # Using row D: x values correspond to columns 1 to 5
        self.play(
            FadeOut(label_0), FadeOut(label_1), FadeOut(label_e), FadeOut(label_pi), FadeOut(label_i),
            e_symbol.animate.move_to(self.grid["D1"]).scale(1/1.2),
            i_unit.animate.move_to(self.grid["D2"]).scale(1/1.2),
            pi.animate.move_to(self.grid["D3"]).scale(1/1.2),
            one.animate.move_to(self.grid["D4"]).scale(1/1.2),
            zero.animate.move_to(self.grid["D5"]).scale(1/1.2),
            run_time=2
        )
        self.wait(2)

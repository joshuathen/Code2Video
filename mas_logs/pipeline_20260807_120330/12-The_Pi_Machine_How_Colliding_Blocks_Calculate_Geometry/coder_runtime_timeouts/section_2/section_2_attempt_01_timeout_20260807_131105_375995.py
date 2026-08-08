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

class Section2Scene(TeachingScene):
    def construct(self):
        title = "The Pattern: Digits of Pi Reveal Themselves"
        lecture_lines = [
            "If M and m are equal, we count 3 clacks.",
            "Increasing M by powers of 100 reveals digits.",
            "Surprisingly, these counts perfectly match the digits of pi."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_COUNTER = "#FFFF00"
        COLOR_PI = "#FFD700"
        COLOR_DIGITS = "#FFFFFF"
        COLOR_M = "#ADD8E6"
        COLOR_m = "#FF69B4"

        # --- Initial Setup for Blocks and Counter ---
        # Wall at the start of Column 3
        wall = Line(self.grid["B3"] + LEFT * 0.5 + UP * 0.5, self.grid["E3"] + LEFT * 0.5 + DOWN * 0.5, color=WHITE)
        ground = Line(self.grid["E3"] + LEFT * 0.5 + DOWN * 0.5, self.grid["E6"] + RIGHT * 0.5 + DOWN * 0.5, color=WHITE)
        
        # Counter setup
        collision_count = ValueTracker(0)
        counter_label = Text("Collisions: ", font_size=24, color=COLOR_COUNTER)
        counter_num = DecimalNumber(0, num_decimal_places=0, color=COLOR_COUNTER)
        counter_num.add_updater(lambda d: d.set_value(collision_count.get_value()))
        counter_group = VGroup(counter_label, counter_num).arrange(RIGHT)
        self.place_at_grid(counter_group, "B4", scale_factor=0.8)

        # Block setup
        block_m = Square(side_length=0.6, color=COLOR_m, fill_opacity=0.5)
        m_label = Text("m", font_size=20, color=COLOR_m)
        block_m_group = VGroup(block_m, m_label)

        block_M = Square(side_length=0.8, color=COLOR_M, fill_opacity=0.5)
        M_label = Text("M", font_size=24, color=COLOR_M)
        block_M_group = VGroup(block_M, M_label)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_COUNTER))
        self.add(wall, ground, counter_group)
        
        # M=1 Case
        self.place_at_grid(block_m_group, "D5")
        self.place_at_grid(block_M_group, "D6")
        self.play(FadeIn(block_m_group), FadeIn(block_M_group))

        # Simulation 1 (Conceptual)
        # Collision 1: M hits m
        self.play(block_M_group.animate.shift(LEFT * 0.5), run_time=0.3)
        collision_count.set_value(1)
        self.play(Flash(block_m_group, color=COLOR_COUNTER), run_time=0.2)
        
        # Collision 2: m hits wall
        self.play(block_m_group.animate.move_to(self.grid["D3"]), run_time=0.4)
        collision_count.set_value(2)
        self.play(Flash(wall, color=COLOR_COUNTER), run_time=0.2)
        
        # Collision 3: m hits M again
        self.play(block_m_group.animate.move_to(self.grid["D4"]), run_time=0.4)
        collision_count.set_value(3)
        self.play(Flash(block_M_group, color=COLOR_COUNTER), run_time=0.2)

        self.wait(1)
        self.play(FadeOut(block_m_group), FadeOut(block_M_group))
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_M))
        
        # M=100 Case
        collision_count.set_value(0)
        M_label_100 = Text("M=100m", font_size=20, color=COLOR_M)
        block_M_100 = Square(side_length=1.0, color=COLOR_M, fill_opacity=0.5)
        block_M_group_100 = VGroup(block_M_100, M_label_100)
        
        self.place_at_grid(block_m_group, "D5")
        self.place_at_grid(block_M_group_100, "D6")
        
        self.play(FadeIn(block_m_group), FadeIn(block_M_group_100))
        # Rapidly increment to 31
        self.play(collision_count.animate.set_value(31), run_time=1.5, rate_func=linear)
        self.wait(1)
        
        # Reset for M=10,000
        self.play(FadeOut(block_m_group), FadeOut(block_M_group_100))
        collision_count.set_value(0)
        
        M_label_10k = Text("M=10,000m", font_size=18, color=COLOR_M)
        block_M_10k = Square(side_length=1.2, color=COLOR_M, fill_opacity=0.5)
        block_M_group_10k = VGroup(block_M_10k, M_label_10k)
        
        self.place_at_grid(block_m_group, "D5")
        self.place_at_grid(block_M_group_10k, "D6")
        
        self.play(FadeIn(block_m_group), FadeIn(block_M_group_10k))
        # Rapidly increment to 314
        self.play(collision_count.animate.set_value(314), run_time=2.0, rate_func=linear)
        self.wait(1)
        self.play(self.lecture[1].animate.set_color(WHITE))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_PI))
        
        # Transition 314 to Pi
        pi_sym = MathTex("\\pi", color=COLOR_PI, font_size=72)
        self.place_at_grid(pi_sym, "B4", scale_factor=1.2)
        
        self.play(
            ReplacementTransform(counter_group, pi_sym),
            FadeOut(block_m_group),
            FadeOut(block_M_group_10k),
            FadeOut(wall),
            FadeOut(ground)
        )
        self.wait(1)
        
        # Flash digits 3.14159...
        pi_digits = Text("3.14159265...", font_size=48, color=COLOR_DIGITS)
        self.place_in_area(pi_digits, "C2", "D5")
        
        self.play(Write(pi_digits))
        self.play(Flash(pi_digits, color=COLOR_DIGITS, line_length=0.4, num_lines=12))
        self.wait(2)

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
        # Data from storyboard
        title = "The Pattern: Digits of Pi Reveal Themselves"
        lecture_lines = [
            "Equal masses result in exactly three clacks.",
            "Increasing the mass ratio reveals powers of ten.",
            "For ten thousand times mass, we get 314 collisions.",
            "The collision counts perfectly match the digits of pi.",
            "This pattern continues as the larger mass grows."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_COUNTER = "#FFFF00"
        COLOR_PI = "#FFD700"
        COLOR_DIGITS = "#FFFFFF"
        COLOR_M = "#ADD8E6"
        COLOR_m = "#FF69B4"

        # --- Constant Elements ---
        # Wall at A3 to F3, Floor at F3 to F6
        # To make it look like a corner, we connect them.
        wall = Line(self.grid["A3"] + UP * 0.5, self.grid["F3"], color=WHITE)
        floor = Line(self.grid["F3"], self.grid["F6"] + RIGHT * 0.5, color=WHITE)
        
        # Counter setup
        collision_count = ValueTracker(0)
        counter_label = Text("Collisions:", font_size=24, color=COLOR_COUNTER)
        counter_num = DecimalNumber(0, num_decimal_places=0, color=COLOR_COUNTER)
        counter_num.add_updater(lambda d: d.set_value(collision_count.get_value()))
        counter_vgroup = VGroup(counter_label, counter_num).arrange(RIGHT, buff=0.2)
        self.place_at_grid(counter_vgroup, "B6", scale_factor=1.0)

        # Block m (small)
        block_m = Square(side_length=0.4, color=COLOR_m, fill_opacity=0.7)
        m_label = Text("m", font_size=16, color=WHITE)
        block_m_full = VGroup(block_m, m_label)

        # Block M (large) - starts with M=m
        block_M = Square(side_length=0.6, color=COLOR_M, fill_opacity=0.7)
        M_label = Text("M", font_size=18, color=WHITE)
        block_M_full = VGroup(block_M, M_label)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_COUNTER))
        self.add(wall, floor, counter_vgroup)
        
        self.place_at_grid(block_m_full, "E4")
        self.place_at_grid(block_M_full, "E5")
        self.play(FadeIn(block_m_full), FadeIn(block_M_full))

        # Collision 1: M hits m
        self.play(block_M_full.animate.shift(LEFT * 0.5), run_time=0.4)
        collision_count.set_value(1)
        self.play(Flash(block_m_full, color=COLOR_COUNTER, flash_radius=0.3), run_time=0.2)

        # Collision 2: m hits wall
        self.play(block_m_full.animate.move_to(self.grid["E3"] + RIGHT * 0.2), run_time=0.4)
        collision_count.set_value(2)
        self.play(Flash(wall, color=COLOR_COUNTER, flash_radius=0.3), run_time=0.2)

        # Collision 3: m hits M
        self.play(block_m_full.animate.move_to(self.grid["E4"]), run_time=0.4)
        collision_count.set_value(3)
        self.play(Flash(block_M_full, color=COLOR_COUNTER, flash_radius=0.3), run_time=0.2)
        
        self.wait(1)
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_M))
        
        # Reset scene for M=100m
        self.play(FadeOut(block_m_full), FadeOut(block_M_full))
        collision_count.set_value(0)
        
        M_label_100 = Text("100m", font_size=16, color=WHITE)
        block_M_100 = VGroup(Square(side_length=0.8, color=COLOR_M, fill_opacity=0.7), M_label_100)
        
        self.place_at_grid(block_m_full, "E4")
        self.place_at_grid(block_M_100, "E6")
        
        self.play(FadeIn(block_m_full), FadeIn(block_M_100))
        
        # Representing 31 collisions
        self.play(
            block_M_100.animate.move_to(self.grid["E4"] + RIGHT * 0.5),
            collision_count.animate.set_value(31),
            run_time=1.5,
            rate_func=linear
        )
        self.wait(1)
        self.play(self.lecture[1].animate.set_color(WHITE))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_M))
        
        # Reset scene for M=10,000m
        self.play(FadeOut(block_m_full), FadeOut(block_M_100))
        collision_count.set_value(0)
        
        M_label_10k = Text("10,000m", font_size=14, color=WHITE)
        block_M_10k = VGroup(Square(side_length=1.0, color=COLOR_M, fill_opacity=0.7), M_label_10k)
        
        self.place_at_grid(block_m_full, "E4")
        self.place_at_grid(block_M_10k, "E6")
        
        self.play(FadeIn(block_m_full), FadeIn(block_M_10k))
        
        # Representing 314 collisions
        self.play(
            block_M_10k.animate.move_to(self.grid["E4"] + RIGHT * 0.5),
            collision_count.animate.set_value(314),
            run_time=2.0,
            rate_func=linear
        )
        self.wait(1)
        self.play(self.lecture[2].animate.set_color(WHITE))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_PI))
        
        # Transform counter 314 to Pi symbol
        pi_symbol = MathTex("\\pi", color=COLOR_PI, font_size=60)
        self.place_at_grid(pi_symbol, "B6")
        
        self.play(
            ReplacementTransform(counter_vgroup, pi_symbol),
            FadeOut(block_m_full),
            FadeOut(block_M_10k),
            FadeOut(wall),
            FadeOut(floor)
        )
        self.wait(1)
        self.play(self.lecture[3].animate.set_color(WHITE))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_DIGITS))
        
        pi_digits = Text("3.14159...", font_size=60, color=COLOR_DIGITS)
        self.place_at_grid(pi_digits, "D4")
        
        self.play(Write(pi_digits))
        self.wait(2)
        self.play(self.lecture[4].animate.set_color(WHITE))

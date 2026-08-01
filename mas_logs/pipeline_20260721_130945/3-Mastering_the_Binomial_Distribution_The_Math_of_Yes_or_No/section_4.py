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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data from shared state
        title_text = "The Binomial Formula Breakdown"
        lecture_lines = [
            "The formula calculates the probability of exactly k successes.",
            "First, count the number of ways to get k successes.",
            "Next, multiply by the probability of those successes occurring.",
            "Then, multiply by the probability of the remaining failures.",
            "Together, these three parts form the complete binomial equation."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Define colors from storyboard
        COLOR_WAYS = "#FFA500" # Orange
        COLOR_SUCCESS = "#0000FF" # Blue
        COLOR_FAILURE = "#FF0000" # Red
        COLOR_MAIN = WHITE

        # Create formula with sub-parts for coloring
        # P(X=k) = (n \choose k) p^k q^{n-k}
        formula = MathTex(
            r"P(X=k)",        # 0
            r"=",             # 1
            r"\binom{n}{k}",  # 2
            r"p^k",           # 3
            r"q^{n-k}",       # 4
            color=COLOR_MAIN
        )
        
        # Placement: Repositioned to avoid overlap with lecture text (Issue 34)
        # Using scale 1.0
        self.place_in_area(formula, "C3", "C6", scale_factor=1.0)

        # Labels - defined here but hidden initially
        # Text mobjects for labels (L022 fallback-safe)
        label_ways = Text("Ways to win", color=COLOR_WAYS)
        label_success = Text("Success probability", color=COLOR_SUCCESS)
        label_failure = Text("Failure probability", color=COLOR_FAILURE)

        # Grid placement for labels relative to formula parts (Issues 35 & 36)
        # Staggered (D4, B5, D6) to prevent horizontal overlap
        self.place_at_grid(label_ways, "D4", scale_factor=0.6)
        self.place_at_grid(label_success, "B5", scale_factor=0.6)
        self.place_at_grid(label_failure, "D6", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        # "The formula calculates the probability of exactly k successes."
        self.play(
            Write(formula),
            self.lecture[0].animate.set_color(COLOR_MAIN),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "First, count the number of ways to get k successes."
        self.play(
            formula[2].animate.set_color(COLOR_WAYS),
            FadeIn(label_ways, shift=UP * 0.2),
            self.lecture[1].animate.set_color(COLOR_WAYS),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Next, multiply by the probability of those successes occurring."
        self.play(
            formula[3].animate.set_color(COLOR_SUCCESS),
            FadeIn(label_success, shift=DOWN * 0.2), # Shift down since label is above
            self.lecture[2].animate.set_color(COLOR_SUCCESS),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Then, multiply by the probability of the remaining failures."
        self.play(
            formula[4].animate.set_color(COLOR_FAILURE),
            FadeIn(label_failure, shift=UP * 0.2),
            self.lecture[3].animate.set_color(COLOR_FAILURE),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Together, these three parts form the complete binomial equation."
        self.play(
            self.lecture[4].animate.set_color(COLOR_MAIN),
            Indicate(formula, color=COLOR_MAIN), # Use Indicate as per L004
            run_time=2
        )
        self.wait(3)

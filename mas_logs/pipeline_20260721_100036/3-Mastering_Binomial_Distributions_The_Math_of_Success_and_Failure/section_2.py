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
        # Setup layout
        title_text = "Defining the Binomial Distribution (The BINS Criteria)"
        lecture_lines = [
            "Binomial distributions follow the BINS criteria.",
            "Binary: Each trial has only two outcomes.",
            "Independent: One trial does not affect another.",
            "Number: The total number of trials is fixed.",
            "Success: The probability of success remains constant."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_BINS = WHITE
        COLOR_LIST = "#ADD8E6"
        
        # Helper functions for icons to avoid asset dependency
        def get_fish():
            # A simple fish-like shape using triangles
            fish_body = Triangle(fill_opacity=1, color=BLUE_B).rotate(-PI/2).scale(0.15)
            fish_tail = Triangle(fill_opacity=1, color=BLUE_B).rotate(PI/2).scale(0.08).next_to(fish_body, LEFT, buff=-0.05)
            return VGroup(fish_body, fish_tail)

        def get_x():
            # An 'X' icon for failure
            return Text("X", color=RED, font_size=36)
        
        # === Animation for Lecture Line 1 ===
        # Binomial distributions follow the BINS criteria.
        self.lecture[0].set_color(COLOR_BINS)
        bins_title = Text("BINS", font_size=72, color=COLOR_BINS)
        self.place_in_area(bins_title, "A2", "A5")
        self.play(Write(bins_title))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Binary: Each trial has only two outcomes.
        self.lecture[1].set_color(COLOR_LIST)
        binary_text = Text("Binary", font_size=32, color=COLOR_LIST)
        # Scaled to 0.4 for consistency and to avoid overlaps
        self.place_at_grid(binary_text, "B2", scale_factor=0.4)
        
        # Prepare slots for the fishing example
        slots = VGroup(*[Square(side_length=0.7, color=WHITE) for _ in range(10)])
        for i, slot in enumerate(slots):
            row_idx = "D" if i < 5 else "E"
            col_idx = str((i % 5) + 1)
            self.place_at_grid(slot, f"{row_idx}{col_idx}", scale_factor=0.8)
            
        self.play(
            FadeIn(binary_text),
            Create(slots)
        )
        
        # Binary outcomes example: Fish (Success) or X (Failure)
        fish_ex = get_fish()
        x_ex = get_x()
        # Resolved Issue 27: Scaled icons to 0.5 to avoid clashing with Row B labels
        self.place_at_grid(fish_ex, "C2", scale_factor=0.5)
        self.place_at_grid(x_ex, "C3", scale_factor=0.5)
        
        self.play(FadeIn(fish_ex), FadeIn(x_ex))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Independent: One trial does not affect another.
        self.lecture[2].set_color(COLOR_LIST)
        independent_text = Text("Independent", font_size=32, color=COLOR_LIST)
        # Resolved Issue 25: Scaled to 0.4 to prevent horizontal overlap
        self.place_at_grid(independent_text, "B3", scale_factor=0.4)
        self.play(FadeIn(independent_text))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # Number: The total number of trials is fixed.
        self.lecture[3].set_color(COLOR_LIST)
        number_text = Text("Number (n=10)", font_size=32, color=COLOR_LIST)
        # Scaled to 0.4 for consistency with Row B labels
        self.place_at_grid(number_text, "B4", scale_factor=0.4)
        self.play(FadeIn(number_text))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # Success: The probability of success remains constant.
        self.lecture[4].set_color(COLOR_LIST)
        same_p_text = Text("Same p", font_size=32, color=COLOR_LIST)
        # Resolved Issue 26: Scaled to 0.4 to prevent horizontal overlap
        self.place_at_grid(same_p_text, "B5", scale_factor=0.4)
        
        # Fill slots sequentially with fishing results
        results = ["F", "X", "F", "F", "X", "F", "X", "F", "X", "F"]
        outcome_group = VGroup()
        for i, res in enumerate(results):
            row_idx = "D" if i < 5 else "E"
            col_idx = str((i % 5) + 1)
            if res == "F":
                m = get_fish()
            else:
                m = get_x()
            # Position inside the squares
            self.place_at_grid(m, f"{row_idx}{col_idx}", scale_factor=0.6)
            outcome_group.add(m)
            
        self.play(FadeIn(same_p_text))
        self.play(LaggedStart(*[FadeIn(m) for m in outcome_group], lag_ratio=0.2))
        self.wait(2)

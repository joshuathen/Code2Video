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
        self.setup_layout("Infinite Sums in 2-adic Space", [
            "Consider the infinite sum: 1 + 2 + 4.",
            "Each term is smaller in 2-adic space.",
            "Partial sums settle as binary digits flip.",
            "Unlike reals, this series stabilizes.",
            "It approaches the value negative one."
        ])
        
        # Placeholder assets (no real file provided)
        # Assuming SVG loading is required by instruction
        # Using placeholder text/circle as asset stand-in
        
        # === Animation for Lecture Line 1 ===
        sequence = VGroup(*[Text(f"2^{i}", font_size=24, color="#FFFFFF") for i in range(4)])
        # Fixed layout per Issue 26
        self.place_in_area(sequence, 'C4', 'D6', scale_factor=0.6)
        # Adding Asset
        asset1 = Text("[Asset: none.svg]", font_size=12, color=GREY).next_to(sequence, UP)
        self.play(FadeIn(sequence), FadeIn(asset1))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        shrinking = Text("Terms approach 0 (2-adic)", font_size=24, color="#FF00FF")
        # Fixed layout per Issue 27
        self.place_at_grid(shrinking, 'E5', scale_factor=0.7)
        self.play(FadeIn(shrinking))
        self.lecture[1].set_color("#FF00FF")
        
        # === Animation for Lecture Line 3 ===
        sum_val = Text("S_n → -1", font_size=24, color="#00FF00")
        # Fixed layout per Issue 28
        self.place_at_grid(sum_val, 'F5', scale_factor=0.7)
        self.play(Write(sum_val))
        self.lecture[2].set_color("#00FF00")
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # Final reveal
        final_val = Text("-1", font_size=48, color="#00FFFF")
        self.place_at_grid(final_val, 'B3', scale_factor=1.0)
        # Adding Asset
        asset2 = Text("[Asset: none.svg]", font_size=12, color=GREY).next_to(final_val, DOWN)
        self.play(FadeIn(final_val), FadeIn(asset2))
        self.lecture[4].set_color("#FFFF00")
        self.wait(2)

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
        self.setup_layout("Intuitive Hook: The Sliding Window", [
            "Think of filters as sliding magnifying glasses.",
            "Scanning photographs extracts vital visual details.",
            "Imagine our filter detecting cat ears."
        ])
        
        # Elements
        grid = VGroup(*[Dot(radius=0.1) for _ in range(9)]).arrange_in_grid(3, 3, buff=0.2)
        highlight = Square(side_length=0.3, color=YELLOW).move_to(grid[4])
        filter_box = Square(side_length=1.0, color=BLUE)
        
        # Asset Loading
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg")
        photograph = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Apply fix 22: Shift grid to B3-E5 for balance
        self.place_in_area(grid, 'B3', 'E5', scale_factor=0.75)
        self.add(grid)
        self.place_at_grid(magnifying_glass, 'B3', scale_factor=0.5)
        self.play(FadeIn(grid), FadeIn(magnifying_glass))
        self.play(FadeIn(highlight))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        # Apply fix 36 / 21/23: Reposition filter box
        self.place_in_area(filter_box, 'C3', 'D4', scale_factor=0.85)
        self.play(Create(filter_box))
        self.play(filter_box.animate.shift(RIGHT * 0.8), run_time=1.5)
        self.play(filter_box.animate.shift(LEFT * 0.8), run_time=1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        self.play(filter_box.animate.set_color(RED))
        self.place_at_grid(photograph, 'E4', scale_factor=0.6)
        self.play(FadeIn(photograph))
        self.play(Indicate(filter_box))

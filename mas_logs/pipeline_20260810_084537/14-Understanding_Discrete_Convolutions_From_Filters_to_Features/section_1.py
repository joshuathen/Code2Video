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
        lecture_lines = [
            "Imagine a magnifying glass scanning a photo.",
            "We call this scan a filter.",
            "It extracts local features like edges.",
            "See how it highlights whiskers?",
            "This is the sliding window."
        ]
        self.setup_layout("Intuitive Hook: The Sliding Window", lecture_lines)
        
        # Load asset
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        
        # Elements
        grid_5x5 = VGroup(*[Square(side_length=0.7, color=WHITE, stroke_width=1) for _ in range(25)])
        grid_5x5.arrange_in_grid(5, 5, buff=0)
        self.place_in_area(grid_5x5, 'B2', 'F6', scale_factor=0.7)
        
        sliding_window = Rectangle(width=0.7*3, height=0.7*3, color=RED, stroke_width=3)
        self.place_at_grid(sliding_window, 'C3', scale_factor=0.7)
        
        # Position glass initially
        self.place_at_grid(magnifying_glass, 'C3', scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(grid_5x5))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(sliding_window), FadeIn(magnifying_glass))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(
            sliding_window.animate.shift(RIGHT * 0.9),
            magnifying_glass.animate.shift(RIGHT * 0.9)
        )
        
        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        green_boundary = Rectangle(width=0.7*3, height=0.7*3, color=GREEN, stroke_width=4)
        green_boundary.move_to(sliding_window.get_center())
        self.play(Create(green_boundary))
        self.play(sliding_window.animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(Indicate(sliding_window), Indicate(magnifying_glass))
        
        self.wait(2)

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
        self.setup_layout("The Setup: The Elastic Collision Paradox", [
            "Two blocks move on a frictionless surface.",
            "A small block approaches a stationary large block.",
            "A wall sits behind the large block."
        ])
        
        # Define assets
        block_small = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#40E0D0")
        block_large = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#FF7F50")
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color="#FFFFFF")
        
        floor = Line(start=self.grid['F1'] + DOWN * 0.5, end=self.grid['F6'] + DOWN * 0.5 + RIGHT * 0.5, color="#FFFFFF")
        
        # Setup initial positions
        self.place_at_grid(block_small, 'E2', scale_factor=0.3)
        self.place_at_grid(block_large, 'E5', scale_factor=0.6)
        self.place_at_grid(wall, 'E6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#40E0D0"))
        self.add(floor)
        self.play(FadeIn(block_small), FadeIn(block_large), FadeIn(wall))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF7F50"))
        self.play(block_small.animate.shift(RIGHT * 1.5))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(Indicate(wall))

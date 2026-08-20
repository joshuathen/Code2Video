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
        self.setup_layout("Introduction: The Unexpected Appearance of π", 
                          ["Count collisions to calculate π?", "Two blocks on a track.", "A simple mechanical puzzle."])
        
        # Assets
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        
        # Elements
        wall = Line(start=UP, end=DOWN, color=GREY).scale(0.5)
        block_a = SVGMobject(asset_path, color="#3498DB", fill_opacity=0.8)
        block_b = SVGMobject(asset_path, color="#E74C3C", fill_opacity=0.8)
        
        label_a = Text("A", font_size=20)
        label_b = Text("B", font_size=20)
        
        # === Animation for Lecture Line 1 ===
        # Count collisions to calculate π?
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(wall, 'D2', scale_factor=0.6)
        self.place_at_grid(block_b, 'D3', scale_factor=0.7)
        label_b.next_to(block_b, UP, buff=0.1).scale(0.7)
        self.add(block_b, label_b)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Two blocks on a track.
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(block_a, 'D5', scale_factor=0.7)
        label_a.next_to(block_a, UP, buff=0.1).scale(0.7)
        self.play(FadeIn(block_a), FadeIn(label_a))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A simple mechanical puzzle.
        self.play(self.lecture[2].animate.set_color(YELLOW))
        # Animate block_a hitting block_b
        self.play(block_a.animate.shift(LEFT * 1.5))
        self.play(
            block_a.animate.set_color(GREEN),
            block_b.animate.set_color(GREEN)
        )
        self.wait(2)

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
            "Can two blocks counting collisions find Pi?",
            "Small mass moves toward a giant wall block.",
            "Collisions occur between these two blocks.",
            "The counts match Pi's first digits!",
            "Let's reveal how this simple math works."
        ]
        self.setup_layout("The Hook: A Counter-Intuitive Challenge", lecture_lines)
        
        # Assets
        block_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        wall = Line(start=UP*2, end=DOWN*2, color=GRAY).shift(RIGHT*1)
        small_block = SVGMobject(block_path, color=WHITE)
        large_block = SVGMobject(block_path, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(small_block, 'F2', scale_factor=0.4)
        small_block.set_color("#FFFFFF")
        self.add(wall, small_block)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        self.place_at_grid(large_block, 'F4', scale_factor=1.0)
        large_block.set_color("#FF0000")
        self.add(large_block)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        # Simulate multiple blocks
        chaos = VGroup(*[SVGMobject(block_path, color="#00FF00").scale(0.2) for _ in range(5)])
        chaos.arrange(RIGHT)
        self.place_in_area(chaos, 'D3', 'E5')
        self.add(chaos)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        self.wait(1)

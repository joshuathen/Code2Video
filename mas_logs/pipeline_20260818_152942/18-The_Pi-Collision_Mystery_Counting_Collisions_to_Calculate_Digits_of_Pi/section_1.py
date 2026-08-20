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
            "Two blocks collide on a frictionless surface.",
            "A small mass hits a massive block.",
            "The wall reflects the smaller block back.",
            "Number of collisions encodes Pi's digits.",
            "The mass difference controls the precision."
        ]
        self.setup_layout("The Impossible Counting Puzzle", lecture_lines)
        
        # Assets
        block_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        wall_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg"
        
        large_block = SVGMobject(block_path, color="#33FF57")
        small_block = SVGMobject(block_path, color="#FF5733")
        wall = SVGMobject(wall_path, color=WHITE)
        
        # Initial placements
        self.place_at_grid(large_block, "C3", scale_factor=1.0)
        self.place_at_grid(small_block, "C2", scale_factor=0.5)
        self.place_at_grid(wall, "C5", scale_factor=1.0)
        
        # ---
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(large_block), FadeIn(small_block), FadeIn(wall))
        self.play(self.lecture[0].animate.set_color("#33FF57"))

        # === Animation for Lecture Line 2 ===
        self.play(small_block.animate.move_to(self.grid["C3"]), run_time=1)
        self.play(self.lecture[1].animate.set_color("#FF5733"))

        # === Animation for Lecture Line 3 ===
        self.play(small_block.animate.move_to(self.grid["C5"]), run_time=1)
        self.play(self.lecture[2].animate.set_color(YELLOW))

        # === Animation for Lecture Line 4 ===
        # Digit placeholder
        digit = Text("3.14", color=WHITE, font_size=36)
        self.place_at_grid(digit, "B3", scale_factor=1.0)
        self.play(FadeIn(digit))
        self.play(self.lecture[3].animate.set_color(ORANGE))

        # === Animation for Lecture Line 5 ===
        self.play(large_block.animate.scale(1.2), small_block.animate.scale(0.8))
        self.play(self.lecture[4].animate.set_color("#FF0000"))
        
        self.wait(2)

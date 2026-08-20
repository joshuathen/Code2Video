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
        self.setup_layout("The Hook: An Impossible Connection", 
                          ["Elastic collisions count digits of Pi.", 
                           "A small block hits a large block.", 
                           "Both bounce between a wall."])
        
        # Define visual elements using Asset
        # Use SVGMobject for the block asset
        block_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        
        wall = Line(start=ORIGIN, end=UP*3, color=GRAY_B)
        small_block = block_asset.copy()
        large_block = block_asset.copy()
        
        # Initial positions adjusted based on feedback
        self.place_at_grid(wall, 'B6', scale_factor=0.8)
        self.place_at_grid(large_block, 'C4', scale_factor=0.7)
        self.place_at_grid(small_block, 'E5', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(wall), FadeIn(large_block), FadeIn(small_block))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        self.play(small_block.animate.move_to(self.grid['C5']))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        halo = Circle(radius=0.5, color=PURPLE, stroke_width=4)
        halo.move_to(self.grid['C4'])
        self.play(ShowPassingFlash(halo, run_time=2))
        self.wait(1)

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
        self.setup_layout("The Physics of Pi: Colliding Blocks", [
            "Collisions calculate the digits of pi.",
            "Mass ratios reveal hidden numerical patterns.",
            "Blocks collide in perfect elastic dance."
        ])
        
        # Assets
        block_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        wall_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg"
        
        # Create mobjects
        block1 = SVGMobject(block_path, color="#FFFFFF")
        label1 = Text("1kg", font_size=18, color=WHITE)
        block2 = SVGMobject(block_path, color="#FFFFFF")
        label2 = Text("100kg", font_size=18, color=WHITE)
        wall = SVGMobject(wall_path, color="#FF4500")
        
        # Grouping
        block1_group = VGroup(block1, label1).arrange(DOWN)
        block2_group = VGroup(block2, label2).arrange(DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(block1_group, "B2", scale_factor=0.5)
        self.place_at_grid(block2_group, "B4", scale_factor=0.8)
        self.play(FadeIn(block1_group), FadeIn(block2_group))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(wall, "E6", scale_factor=0.6)
        ratio_text = Text("Mass Ratio: 1:100", font_size=20, color="#FF4500")
        self.place_at_grid(ratio_text, "E2")
        self.play(FadeIn(wall), FadeIn(ratio_text))
        self.play(block2_group.animate.next_to(wall, LEFT, buff=0.1))
        self.lecture[1].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        collision_block = SVGMobject(block_path, color="#00FF00")
        self.place_at_grid(collision_block, "D4", scale_factor=0.5)
        self.play(FadeIn(collision_block))
        self.play(collision_block.animate.shift(LEFT * 0.5), run_time=0.5)
        self.play(collision_block.animate.shift(RIGHT * 0.5), run_time=0.5)
        self.lecture[2].set_color("#00FF00")
        self.wait(1)

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
        self.setup_layout("The Hook: An Unexpected Connection", [
            "Colliding blocks can calculate digits of Pi.",
            "Tiny block bounces off a massive block.",
            "The number of collisions matches Pi digits."
        ])
        
        # Load assets
        block_small = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#FF5733")
        block_large = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#33FF57")
        
        # Group them
        block_group = VGroup(block_small, block_large).arrange(RIGHT, buff=0.2)
        self.place_in_area(block_group, 'B4', 'D6', scale_factor=0.8)
        
        label = Text("Unexpected Connection", font_size=24, color=WHITE)
        self.place_at_grid(label, 'B5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(block_group), FadeIn(label))
        self.lecture[0].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visualizing tiny block bouncing off massive block
        self.play(block_small.animate.shift(LEFT * 1.5))
        self.play(block_small.animate.shift(RIGHT * 1.5))
        
        self.lecture[1].set_color("#FFC300")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visualizing a single collision
        self.play(
            block_small.animate.set_x(block_large.get_x() - 0.4),
            run_time=0.5
        )
        self.play(
            block_small.animate.set_x(block_small.get_x() - 1.0),
            run_time=0.5
        )
        
        self.lecture[2].set_color("#33FF57")
        self.wait(2)

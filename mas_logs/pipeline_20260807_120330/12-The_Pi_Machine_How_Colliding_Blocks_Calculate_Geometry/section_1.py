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
        # Lecture lines from storyboard
        lecture_lines = [
            "Imagine two blocks and a wall on frictionless ice.",
            "A small block 'm' sits near the wall.",
            "A massive block 'M' slides toward the small one."
        ]
        self.setup_layout("The Setup: An Unlikely Experiment", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(YELLOW)
        
        # Wall at Column 3 (A3 to F3) as per Storyboard
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg
        wall_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg"
        wall = SVGMobject(wall_path)
        wall.set_color(WHITE)
        # Scale to span the height of the grid from Row A to Row F
        wall.height = 5.0
        self.place_in_area(wall, "A3", "F3")
        
        # Floor from Column 3 to Column 6
        # Align floor with the bottom of the wall (Row F)
        floor_start = self.grid["F3"] + DOWN * 0.5 + LEFT * 0.5
        floor_end = self.grid["F6"] + DOWN * 0.5 + RIGHT * 0.5
        floor = Line(floor_start, floor_end, color=WHITE)
        
        self.play(DrawBorderThenFill(wall), Create(floor))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Small block 'm' at E4 (near wall)
        # Applying Critic scale 1.2 (Issue 21)
        m_block = Square(side_length=0.5, fill_opacity=1, color=BLUE)
        self.place_at_grid(m_block, "E4", scale_factor=1.2)
        # Label m positioned 0.1 units above the block
        m_label = MathTex("m", color=WHITE, font_size=32).next_to(m_block, UP, buff=0.1)
        
        self.play(FadeIn(m_block), Write(m_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Massive block 'M' at E6
        # Applying Critic scale 2.0 (Issue 22, 23)
        M_block = Square(side_length=0.5, fill_opacity=1, color=RED)
        self.place_at_grid(M_block, "E6", scale_factor=2.0)
        # Label M positioned 0.1 units above the block
        M_label = MathTex("M", color=WHITE, font_size=40).next_to(M_block, UP, buff=0.1)
        
        # Group M and its label to move together
        M_group = VGroup(M_block, M_label)
        
        # Counter at B6 for collision counting
        clack_count = Integer(0, color=YELLOW).scale(1.2)
        self.place_at_grid(clack_count, "B6")
        clack_label = Text("Clacks:", font_size=20, color=YELLOW).next_to(clack_count, LEFT, buff=0.2)
        
        self.play(FadeIn(M_group), FadeIn(clack_label), FadeIn(clack_count))
        self.wait(0.5)
        
        # Calculate key positions for collisions
        # Wall is at x=2.5 (Column 3)
        # m_block size = 0.5 * 1.2 = 0.6. Half-width = 0.3.
        # M_block size = 0.5 * 2.0 = 1.0. Half-width = 0.5.
        
        # Hit 1: M slides left to hit m. m is centered at E4 (x=3.5).
        # Collision occurs when x_M - 0.5 = x_m + 0.3 => x_M = 3.5 + 0.8 = 4.3.
        dist_M_move = self.grid["E6"][0] - 4.3
        self.play(M_group.animate.shift(LEFT * dist_M_move), run_time=1.5)
        
        clack_count.set_value(1)
        self.play(Flash(clack_count, color=YELLOW, flash_radius=0.2))
        
        # Hit 2: m flies left to hit wall (x=2.5).
        # Collision occurs when x_m - 0.3 = 2.5 => x_m = 2.8.
        dist_m_to_wall = self.grid["E4"][0] - 2.8
        self.play(
            m_block.animate.shift(LEFT * dist_m_to_wall),
            m_label.animate.shift(LEFT * dist_m_to_wall),
            run_time=0.6
        )
        
        clack_count.set_value(2)
        self.play(Flash(clack_count, color=YELLOW, flash_radius=0.2))
        
        # m rebounds slightly to the right to separate from wall
        self.play(
            m_block.animate.shift(RIGHT * 0.4),
            m_label.animate.shift(RIGHT * 0.4),
            run_time=0.6
        )
        
        self.wait(1)
        self.lecture[2].set_color(WHITE)
        self.wait(2)

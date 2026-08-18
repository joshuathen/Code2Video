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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Observe the mass ratio of the two blocks.",
            "For ratio powers of one hundred, count the collisions.",
            "Impacts match digits of Pi sequentially.",
            "Pi emerges from pure mechanical interaction.",
            "A stunning connection between physics and numbers."
        ]
        self.setup_layout("The Bridge: Collision Counting and Pi", lecture_lines)
        
        # Colors for each lecture line
        colors = [BLUE_C, GREEN_C, YELLOW_C, RED_C, PURPLE_C]
        
        # Assets
        wall_icon = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg"
        block_icon = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"

        # === Animation for Lecture Line 1 ===
        # Represent the collision as a point hitting a wall.
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg]
        point = Dot(color=WHITE)
        wall = SVGMobject(wall_icon, color=GREY)
        self.place_at_grid(point, 'B2', scale_factor=0.8)
        self.place_at_grid(wall, 'C6', scale_factor=0.5)
        self.add(point, wall)
        self.lecture[0].set_color(colors[0])
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Add a second boundary to capture multiple bounces using a wall.
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg]
        wall2 = SVGMobject(wall_icon, color=GREY)
        self.place_at_grid(wall2, 'C3', scale_factor=0.5)
        self.add(wall2)
        self.lecture[1].set_color(colors[1])
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visualize the path as a sequence of segments hitting a block.
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg]
        block = SVGMobject(block_icon, color=ORANGE)
        self.place_at_grid(block, 'C4', scale_factor=0.5)
        path = VGroup(
            Line(self.grid['C3'], self.grid['C6'], color=ORANGE),
            Line(self.grid['C6'], self.grid['C3'], color=ORANGE)
        )
        self.add(path, block)
        self.lecture[2].set_color(colors[2])
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Count the number of collisions against the walls.
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg]
        collision_text = Text("Collisions: 3, 31, 314", font_size=24, color=WHITE)
        self.place_in_area(collision_text, 'C3', 'D4', scale_factor=0.7)
        self.add(collision_text)
        self.lecture[3].set_color(colors[3])
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Relate the total bounce count against the block to digits of Pi.
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg]
        pi_text = MathTex(r"\\pi \\approx 3.14159...", color=YELLOW)
        self.place_in_area(pi_text, 'E3', 'F4', scale_factor=0.7)
        self.add(pi_text)
        self.lecture[4].set_color(colors[4])
        self.wait(2)

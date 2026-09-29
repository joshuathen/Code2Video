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
        self.setup_layout("The Hook: A Surprising Connection", [
            "Collisions between two blocks can compute Pi.",
            "Small mass m hits large mass M.",
            "M is much larger than m.",
            "A wall reflects the small block back.",
            "[Asset: CollisionCounter] counts every impact."
        ])
        
        # Define mobjects using provided assets
        m_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#00FF00")
        M_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#00FF00")
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color="#FFFFFF")
        
        # Initial arrangement
        self.place_at_grid(m_block, "C3", scale_factor=0.3)
        self.place_at_grid(M_block, "C5", scale_factor=0.6)
        self.place_at_grid(wall, "C2", scale_factor=0.5)

        # Group for animation positioning
        animation_group = VGroup(m_block, M_block, wall)
        self.place_in_area(animation_group, 'C3', 'E5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(Create(m_block), Create(M_block))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(m_block.animate.set_color("#FFFF00"), run_time=1)
        self.play(m_block.animate.next_to(M_block, LEFT), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        self.play(Create(wall))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        counter = Text("Collisions: 1", font_size=24, color="#00FF00")
        self.place_at_grid(counter, 'D3', scale_factor=0.7)
        self.add(counter)
        self.wait(2)

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
            "Two blocks collide: small mass m, large mass M.",
            "They bounce between a wall and each other.",
            "For M=100m, we see exactly 31 collisions.",
            "Surprisingly, this counts the digits of pi.",
            "Let's discover why this happens."
        ]
        self.setup_layout("The Unexpected Connection", lecture_lines)
        
        # Assets
        block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        
        # Group assets
        collision_group = VGroup(block, wall)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#F8F9FA")
        self.place_at_grid(block, 'D2', scale_factor=0.6)
        self.place_at_grid(wall, 'D4', scale_factor=0.7)
        self.play(FadeIn(block), FadeIn(wall))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#E9ECEF")
        line = Line(start=block.get_center(), end=wall.get_center(), color="#E9ECEF")
        self.play(Create(line))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFC107")
        self.place_in_area(collision_group, 'D2', 'E5', scale_factor=0.8)
        flash = Dot(color="#FFC107").move_to(collision_group.get_center()).scale(2)
        self.play(Flash(flash, color="#FFC107", flash_radius=0.5))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#ADB5BD")
        self.play(FadeOut(block), FadeOut(wall), FadeOut(line), FadeOut(flash))

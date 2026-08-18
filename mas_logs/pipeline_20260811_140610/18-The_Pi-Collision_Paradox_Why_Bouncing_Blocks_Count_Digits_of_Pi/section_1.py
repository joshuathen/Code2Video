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
        self.setup_layout("The Hook: An Impossible Counting Puzzle", [
            "Two blocks slide on a frictionless surface.",
            "One block moves toward a wall-bound block.",
            "How many collisions happen before they stop?"
        ])
        
        # Assets
        m1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#FF00FF")
        m2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#00FFFF")
        
        m1_label = Text("m1", color="#FF00FF", font_size=24).scale(0.7)
        m2_label = Text("m2", color="#00FFFF", font_size=24).scale(0.7)
        
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color=WHITE)
        wall_label = Text("Wall", color=WHITE, font_size=20).scale(0.7)
        collision_counter = Text("Collisions: 0", color=WHITE, font_size=24).scale(0.8)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(m1, 'B4', scale_factor=0.8)
        self.place_at_grid(m2, 'B5', scale_factor=0.8)
        
        m1_label.next_to(m1, UP, buff=0.1)
        m2_label.next_to(m2, UP, buff=0.1)
        
        self.add(m1, m2, m1_label, m2_label)
        self.play(FadeIn(m1), FadeIn(m2), Write(m1_label), Write(m2_label))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        self.place_in_area(wall, 'B5', 'B6', scale_factor=0.6)
        self.place_at_grid(wall_label, 'A5', scale_factor=0.7)
        
        self.play(m1.animate.shift(LEFT * 0.5))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(collision_counter, 'D4', scale_factor=0.8)
        self.play(Write(collision_counter))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)

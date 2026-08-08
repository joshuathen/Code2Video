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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Add a wall behind the smaller block.",
            "The block bounces between the wall and big block.",
            "A counter tracks the total number of collisions."
        ]
        self.setup_layout("Visualizing the Collision Count", lecture_lines)
        
        # Load asset and objects
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        wall.set_color("#AAAAAA")
        self.place_at_grid(wall, "E2", scale_factor=0.8)
        
        small_block = Circle(radius=0.3, color=BLUE, fill_opacity=0.8)
        self.place_at_grid(small_block, "E3", scale_factor=0.8)
        
        big_block = Circle(radius=0.6, color=RED, fill_opacity=0.8)
        self.place_at_grid(big_block, "E5", scale_factor=0.8)
        
        counter_label = Text("Collisions:", font_size=24)
        counter_val = Integer(0, font_size=24, color=WHITE)
        counter = VGroup(counter_label, counter_val).arrange(RIGHT)
        self.place_in_area(counter, "B4", "B6", scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(wall))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        # Bouncing animation path
        path = VMobject()
        path.set_points_smoothly([self.grid["E3"], self.grid["E4"], self.grid["E3"]])
        self.play(MoveAlongPath(small_block, path), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.play(FadeIn(counter))
        
        # Update counter simulation
        current_counter_val = 0
        for i in range(1, 4):
            new_val = Integer(i, font_size=24, color=YELLOW)
            new_val.move_to(counter_val.get_center())
            self.play(Transform(counter_val, new_val), run_time=0.5)
            self.wait(0.2)
        self.wait(1)
